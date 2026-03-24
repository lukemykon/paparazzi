/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/MAV_fast_controller_group12_cmjong/MAV_fast_controller_group12_cmjong.c"
 * @author Roland Meertens
 *
 * Obstacle avoidance controller for AE4317 Autonomous Flight of MAVs (TU Delft).
 * Uses colour detection (cv_detect_color_object) and optical flow to detect
 * obstacles and navigate safely inside the CyberZoo arena.
 *
 * State machine:
 *   SAFE_AND_WAIT  ──no obstacle──►  MOVE_FORWARD_WITH_FIXED_DISTANCE
 *   SAFE_AND_WAIT  ──obstacle──────►  TURN_AVOID
 *   TURN_AVOID     ──clear──────────►  SAFE_AND_WAIT
 *   MOVE_FORWARD   ──obstacle──────►  SAFE_AND_WAIT
 *   MOVE_FORWARD   ──out of zone───►  OUT_OF_BOUNDS
 *   OUT_OF_BOUNDS  ──back inside───►  SAFE_AND_WAIT
 */

#include "modules/MAV_fast_controller_group12_cmjong/MAV_fast_controller_group12_cmjong.h"
#include "modules/computer_vision/MAV_cv_detect_group12_cmjong.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include "mcu_periph/sys_time.h"
#include <time.h>
#include <stdio.h>
#include <math.h>

#include "generated/flight_plan.h"

/* ---------- debug logging ------------------------------------------------ */

#define MAV_FAST_CONTROLLER_VERBOSE TRUE

#define PRINT(string, ...) \
  fprintf(stderr, "[MAV_fast_controller->%s()] " string, __FUNCTION__, ##__VA_ARGS__)

#if MAV_FAST_CONTROLLER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

/* ---------- tuning constants --------------------------------------------- */

/** Incremental heading change per avoidance tick (degrees). */
#define AVOIDANCE_TURN_DEG          5.f

/** Incremental heading change when recovering from out-of-bounds (degrees). */
#define OOB_TURN_DEG                5.f

/** Total heading rotation triggered by an optical-flow obstacle (degrees). */
#define OF_AVOIDANCE_TOTAL_DEG      100.f

/** Distance moved forward per MOVE_FORWARD tick (metres). */
#define MOVE_STEP_M                 0.5f

/** Look-ahead distance used when probing the next waypoint position (metres). */
#define OOB_PROBE_DISTANCE_M        1.5f

/** Body yaw-rate (rad/s) above which new OF detections are suppressed. */
#define GYRO_YAW_RATE_THRESHOLD     0.15f

/** Seconds after take-off during which OF detections are ignored. */
#define OF_STARTUP_IGNORE_S         2.0f

/** brake before turning */
static bool did_backward_step = true;
#define BACKWARD_STEP_M         0.10f

/* ---------- state machine ------------------------------------------------ */

/** Navigation states.
 *  NOTE: GATE_DETECTED is reserved for future use and is not yet handled. */
typedef enum {
  SAFE_AND_WAIT,
  BRAKE_BEFORE_TURN,   
  TURN_AVOID,
  SEARCH_FOR_SAFE_HEADING,
  MOVE_FORWARD_WITH_FIXED_DISTANCE,
  GATE_DETECTED,
  OUT_OF_BOUNDS,
} navigation_state_t;

/* ---------- module state ------------------------------------------------- */

static navigation_state_t navigation_state = SAFE_AND_WAIT;

/** True when the optical-flow module reports an obstacle ahead. */
static bool of_obstacle_ahead = false;

/** Remaining degrees of the current OF-triggered rotation. */
static float of_turn_remaining = 0.f;

/** True once the drone has taken off for the first time this session. */
static bool was_in_flight = false;

/** Absolute time (s) until which OF detections are suppressed after take-off. */
static float of_ignore_until = 0.f;

/* Colour detection results (updated via ABI callback). */
static uint16_t detected_local           = 1;
static uint16_t color_count_orange_local = 0;
static uint16_t color_count_green_local  = 0;
static uint16_t color_count_blue_local   = 0;

/* ---------- ABI bindings ------------------------------------------------- */

#ifndef MAV_cmjong_VISUAL_DETECTION_ID
#define MAV_cmjong_VISUAL_DETECTION_ID    ABI_BROADCAST
#endif
#ifndef MAV_cmjong_OF_VISUAL_DETECTION_ID
#define MAV_cmjong_OF_VISUAL_DETECTION_ID ABI_BROADCAST
#endif

static abi_event cv_detect_event;
static abi_event luke_of_event;

/**
 * @brief ABI callback: colour detection result from cv_detect_color_object.
 *
 * @param detected        Non-zero if any obstacle colour was detected.
 * @param color_count_*   Pixel counts for each tracked colour band.
 */
static void cv_detection_message_callback(
    uint8_t  __attribute__((unused)) sender_id,
    int16_t  detected,
    int16_t  color_count_orange,
    int16_t  color_count_green,
    int16_t  color_count_blue,
    int32_t  __attribute__((unused)) extra2,
    int16_t  __attribute__((unused)) extra3)
{
  detected_local           = (uint16_t)detected;
  color_count_orange_local = (uint16_t)color_count_orange;
  color_count_green_local  = (uint16_t)color_count_green;
  color_count_blue_local   = (uint16_t)color_count_blue;
}

/**
 * @brief ABI callback: optical-flow obstacle quality from the OF module.
 *
 * A positive quality value signals that the OF divergence exceeds the
 * obstacle threshold.
 */
static void luke_of_message_callback(
    uint8_t  __attribute__((unused)) sender_id,
    int16_t  __attribute__((unused)) pixel_x,
    int16_t  __attribute__((unused)) pixel_y,
    int16_t  __attribute__((unused)) pixel_width,
    int16_t  __attribute__((unused)) pixel_height,
    int32_t  quality,
    int16_t  __attribute__((unused)) extra)
{
  of_obstacle_ahead = (quality > 0);
}

/* ---------- helper prototypes -------------------------------------------- */

static void    rotate_drone_heading(float degrees);
static void    move_waypoint_forward(uint8_t waypoint, float distance_m);
static void    calculate_forward_position(struct EnuCoor_i *new_coor, float distance_m);
static void    set_waypoint_position(uint8_t waypoint, struct EnuCoor_i *new_coor);
static void    hold_current_waypoints(void);
static bool    any_obstacle_detected(void);

/* ---------- public API --------------------------------------------------- */

/** @brief One-time initialisation: bind ABI message handlers. */
void MAV_fast_controller_group12_cmjong_init(void)
{
  AbiBindMsgVISUAL_DETECTION(MAV_cmjong_VISUAL_DETECTION_ID,
                              &cv_detect_event,
                              cv_detection_message_callback);
  AbiBindMsgVISUAL_DETECTION(MAV_cmjong_OF_VISUAL_DETECTION_ID,
                              &luke_of_event,
                              luke_of_message_callback);
}

/**
 * @brief Periodic control loop – runs at the module update rate.
 *
 * Evaluates the navigation state machine and issues waypoint / heading
 * commands to the autopilot.  Does nothing while the drone is on the ground.
 */
void MAV_fast_controller_group12_cmjong_periodic(void)
{
  /* Only run while airborne. */
  if (!autopilot_in_flight()) {
    was_in_flight      = false;
    navigation_state   = SAFE_AND_WAIT;
    of_obstacle_ahead  = false;
    of_turn_remaining  = 0.f;
    return;
  }

  const float now = get_sys_time_float();

  /* One-shot initialisation on first airborne tick. */
  if (!was_in_flight) {
    was_in_flight         = true;
    of_ignore_until       = now + OF_STARTUP_IGNORE_S;
    of_obstacle_ahead     = false;
    of_turn_remaining     = 0.f;
    luke_of_request_reset = true;
  }

  /* Suppress OF detections during start-up grace period. */
  if (now < of_ignore_until) {
    of_obstacle_ahead = false;
  }

  /* Suppress OF detections while the drone is actively yawing, to avoid
   * false positives from rotational optical flow. */
  if (fabsf(stateGetBodyRates_f()->r) > GYRO_YAW_RATE_THRESHOLD) {
    of_obstacle_ahead = false;
  }

  VERBOSE_PRINT(
    "State: %d | detected: %u (%u orange, %u green, %u blue) "
    "| of: %d | of_rem: %.1f | grace: %.1f\n",
    navigation_state,
    detected_local,
    color_count_orange_local,
    color_count_green_local,
    color_count_blue_local,
    (int)of_obstacle_ahead,
    of_turn_remaining,
    fmaxf(0.f, of_ignore_until - now)
  );

  /* ---- State machine ---------------------------------------------------- */
  switch (navigation_state) {

    /* Hold position until the sensors give a clean reading. */
    case SAFE_AND_WAIT:
      hold_current_waypoints();

      if (!any_obstacle_detected()) {
        navigation_state = MOVE_FORWARD_WITH_FIXED_DISTANCE;
      } else {
        /* Prime the OF rotation budget only on fresh OF triggers. */
        if (of_obstacle_ahead && detected_local == 0 && of_turn_remaining == 0.f) {
          of_turn_remaining = OF_AVOIDANCE_TOTAL_DEG;
        }
        navigation_state = BRAKE_BEFORE_TURN;
      }
      break;
    
      case BRAKE_BEFORE_TURN:

        /* Step 1: stop movement */
        hold_current_waypoints();

        /* Step 2: move slightly backward once */
        if (!did_backward_step) {
          move_waypoint_forward(WP_TRAJECTORY, -BACKWARD_STEP_M);
          move_waypoint_forward(WP_GOAL,       -BACKWARD_STEP_M);
          did_backward_step = true;
          break;
        }

        /* Step 3: now safe to turn */
        navigation_state = TURN_AVOID;
        break;

    /* Rotate until obstacles are clear. */
    case TURN_AVOID:
      if (of_turn_remaining > 0.f) {
        /* Consume the OF rotation budget in fixed increments. */
        const float step = fminf(of_turn_remaining, AVOIDANCE_TURN_DEG);
        rotate_drone_heading(step);
        PRINT("Turned: %0.1f", step);
        of_turn_remaining -= step;

        if (of_turn_remaining <= 0.f) {
          of_turn_remaining     = 0.f;
          luke_of_request_reset = true;
          navigation_state      = SAFE_AND_WAIT;
        }
      } else if (of_obstacle_ahead) {
        /* New OF detection while not already rotating: start a fresh budget. */
        of_turn_remaining = OF_AVOIDANCE_TOTAL_DEG - AVOIDANCE_TURN_DEG;
        rotate_drone_heading(AVOIDANCE_TURN_DEG);
        PRINT("Turned: %0.1f", AVOIDANCE_TURN_DEG);
      } else if (detected_local > 0) {
        /* Colour obstacle: small incremental turn. */
        rotate_drone_heading(AVOIDANCE_TURN_DEG);
        PRINT("Turned: %0.1f", AVOIDANCE_TURN_DEG);
      } else {
        navigation_state = SAFE_AND_WAIT;
      }

      break;

    /* Advance the drone by one fixed step if the path is clear. */
    case MOVE_FORWARD_WITH_FIXED_DISTANCE: {
      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY),
                              WaypointY(WP_TRAJECTORY))) {
        navigation_state = OUT_OF_BOUNDS;
        break;
      }

      if (any_obstacle_detected()) {
        navigation_state = SAFE_AND_WAIT;
        break;
      }

      struct EnuCoor_i next_coor;
      calculate_forward_position(&next_coor, MOVE_STEP_M);

      if (!InsideObstacleZone(POS_FLOAT_OF_BFP(next_coor.x),
                              POS_FLOAT_OF_BFP(next_coor.y))) {
        navigation_state = OUT_OF_BOUNDS;
      } else {
        set_waypoint_position(WP_TRAJECTORY, &next_coor);
        set_waypoint_position(WP_GOAL,       &next_coor);
      }
      break;
    }

    /* Slowly rotate and probe until the drone is back inside the arena. */
    case OUT_OF_BOUNDS:
      rotate_drone_heading(OOB_TURN_DEG);
      move_waypoint_forward(WP_TRAJECTORY, OOB_PROBE_DISTANCE_M);

      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY),
                             WaypointY(WP_TRAJECTORY))) {
        navigation_state = SAFE_AND_WAIT;
      }
      break;

    /* Future states – fall through to avoid undefined behaviour. */
    case SEARCH_FOR_SAFE_HEADING:
    case GATE_DETECTED:
    default:
      PRINT("Unhandled navigation state %d – reverting to SAFE_AND_WAIT\n",
            navigation_state);
      navigation_state = SAFE_AND_WAIT;
      break;
  }
}

/* ---------- helpers ------------------------------------------------------- */

/**
 * @brief Returns true if either the colour filter or the OF module reports
 *        an obstacle this tick.
 */
static bool any_obstacle_detected(void)
{
  return (detected_local > 0) || of_obstacle_ahead;
}

/**
 * @brief Rotate the drone's desired heading by @p degrees (positive = CCW).
 *
 * Heading is normalised to [-π, π] before being written to nav.heading.
 */
static void rotate_drone_heading(float degrees)
{
  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(degrees);
  FLOAT_ANGLE_NORMALIZE(new_heading);
  nav.heading = new_heading;
}

/**
 * @brief Compute the ENU position @p distance_m ahead of the current pose
 *        along the current heading and write it to @p new_coor.
 */
static void calculate_forward_position(struct EnuCoor_i *new_coor, float distance_m)
{
  const float heading = stateGetNedToBodyEulers_f()->psi;
  new_coor->x = stateGetPositionEnu_i()->x
                + POS_BFP_OF_REAL(sinf(heading) * distance_m);
  new_coor->y = stateGetPositionEnu_i()->y
                + POS_BFP_OF_REAL(cosf(heading) * distance_m);
}

/**
 * @brief Move @p waypoint to the position @p distance_m ahead of the drone.
 */
static void move_waypoint_forward(uint8_t waypoint, float distance_m)
{
  struct EnuCoor_i new_coor;
  calculate_forward_position(&new_coor, distance_m);
  set_waypoint_position(waypoint, &new_coor);
}

/**
 * @brief Move @p waypoint to the given ENU integer coordinates.
 */
static void set_waypoint_position(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
}

/**
 * @brief Snap both GOAL and TRAJECTORY waypoints to the current drone position.
 *
 * Called while in SAFE_AND_WAIT to prevent the autopilot from driving the
 * drone toward a stale waypoint while the controller is assessing the scene.
 */
static void hold_current_waypoints(void)
{
  waypoint_move_here_2d(WP_GOAL);
  waypoint_move_here_2d(WP_TRAJECTORY);
}