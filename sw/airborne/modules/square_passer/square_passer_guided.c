#include "modules/square_passer/square_passer_guided.h"

#include "autopilot.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/flight_plan.h"
#include "modules/core/abi.h"
#include "modules/core/abi_sender_ids.h"
#include "mcu_periph/sys_time.h"
#include "state.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SQUARE_PASSER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[square_passer_guided->%s()] " string, __FUNCTION__ , ##__VA_ARGS__)
#if SQUARE_PASSER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

enum square_passer_state_t {
  SP_SEARCH,
  SP_TRACK,
  SP_PASS_THROUGH
};

float spg_forward_speed = 0.55f;
float spg_max_lateral_speed = 0.40f;
float spg_lateral_gain = 0.85f;
float spg_search_heading_rate = RadOfDeg(36.f);
float spg_search_forward_speed = 0.25f;
float spg_center_forward_speed = 0.40f;
float spg_center_confirm_time = 0.45f;
float spg_vertical_gain = 1.2f;
float spg_max_vertical_step = 0.25f;
float spg_vertical_target_ratio = -0.10f;
float spg_search_vertical_rate = 0.25f;
float spg_search_min_altitude = 0.9f;
float spg_search_max_altitude = 1.30f;
float spg_detection_timeout = 2.0f;
float spg_gate_pass_distance = 1.60f;
float spg_gate_center_tolerance = 0.12f;
float spg_gate_vertical_tolerance = 0.12f;
float spg_pass_through_time = 1.4f;
float spg_min_guided_altitude = 0.9f;
float spg_camera_delay_comp_s = 0.16f;
float spg_max_prediction_horizon_s = 0.45f;
float spg_max_relative_velocity = 3.0f;

static enum square_passer_state_t sp_state = SP_SEARCH;
static bool sp_enabled = false;
static float sp_turn_direction = 1.f;
static float sp_last_detection_s = -1000.f;
static float sp_pass_until_s = 0.f;
static float sp_last_periodic_s = -1.f;
static float sp_centered_since_s = -1.f;
static float sp_track_seen_since_s = -1.f;
static float sp_candidate_seen_since_s = -1.f;
static float sp_search_altitude = 1.1f;
static float sp_search_alt_dir = 1.f;
static float sp_search_turn_remaining_deg = 360.f;
static float sp_search_step_remaining_m = 0.f;
static float sp_search_anchor_x = 0.f;
static float sp_search_anchor_y = 0.f;
static bool sp_search_anchor_valid = false;

static float sp_gate_x = -10.f;
static float sp_gate_y = 0.f;
static float sp_gate_z = 0.f;
static float sp_track_lost_grace_s = 1.5f;
static float sp_max_track_distance = 7.0f;
static float sp_lock_confirm_time = 0.12f;
static float sp_search_scan_turn_deg = 220.f;
static float sp_search_boundary_turn_deg = 120.f;
static float sp_search_step_distance = 2.8f;
static float sp_post_pass_cooldown_s = 2.2f;
static float sp_pass_exit_no_gate_s = 0.9f;
static float sp_pass_completed_at_s = -100.f;

// Memory of last gate bearing for smarter search
static float sp_last_gate_bearing_deg = 0.f;
static bool sp_gate_bearing_valid = false;

// Heading at detection time for latency compensation
static float sp_gate_heading_at_detection = 0.f;
static float sp_gate_x_raw = -10.f;
static float sp_gate_y_raw = 0.f;
static float sp_gate_z_raw = 0.f;
static float sp_gate_vx_raw = 0.f;
static float sp_gate_vy_raw = 0.f;
static float sp_gate_vz_raw = 0.f;
static bool sp_gate_velocity_initialized = false;
static float sp_prev_meas_x = -10.f;
static float sp_prev_meas_y = 0.f;
static float sp_prev_meas_z = 0.f;
static float sp_prev_meas_s = -1.f;

// EMA smoothing for gate position (reduces frame-to-frame jitter)
static float sp_gate_ema_alpha = 0.6f;  // blend factor: higher = more responsive, lower = smoother
static bool sp_gate_ema_initialized = false;

#ifndef SQUARE_PASSER_RELATIVE_LOCALIZATION_ID
#define SQUARE_PASSER_RELATIVE_LOCALIZATION_ID DETECT_GATE_ABI_ID
#endif

static abi_event relative_localization_ev;
static float clampf(float value, float min_value, float max_value);

static void relative_localization_cb(uint8_t __attribute__((unused)) sender_id,
                                     int32_t __attribute__((unused)) id,
                                     float x, float y, float z,
                                     float __attribute__((unused)) vx,
                                     float __attribute__((unused)) vy,
                                     float __attribute__((unused)) vz)
{
  float now = get_sys_time_float();

  // Estimate relative gate velocity from measurement deltas to compensate camera/processing delay.
  if (sp_prev_meas_s > 0.f) {
    float dt = now - sp_prev_meas_s;
    if (dt > 0.03f && dt < 0.60f) {
      float vx_est = (x - sp_prev_meas_x) / dt;
      float vy_est = (y - sp_prev_meas_y) / dt;
      float vz_est = (z - sp_prev_meas_z) / dt;

      vx_est = clampf(vx_est, -spg_max_relative_velocity, spg_max_relative_velocity);
      vy_est = clampf(vy_est, -spg_max_relative_velocity, spg_max_relative_velocity);
      vz_est = clampf(vz_est, -spg_max_relative_velocity, spg_max_relative_velocity);

      if (!sp_gate_velocity_initialized) {
        sp_gate_vx_raw = vx_est;
        sp_gate_vy_raw = vy_est;
        sp_gate_vz_raw = vz_est;
        sp_gate_velocity_initialized = true;
      } else {
        const float vel_alpha = 0.55f;
        sp_gate_vx_raw = vel_alpha * vx_est + (1.f - vel_alpha) * sp_gate_vx_raw;
        sp_gate_vy_raw = vel_alpha * vy_est + (1.f - vel_alpha) * sp_gate_vy_raw;
        sp_gate_vz_raw = vel_alpha * vz_est + (1.f - vel_alpha) * sp_gate_vz_raw;
      }
    }
  }
  sp_prev_meas_x = x;
  sp_prev_meas_y = y;
  sp_prev_meas_z = z;
  sp_prev_meas_s = now;

  // Store raw detection in body frame at capture time, with EMA smoothing
  if (!sp_gate_ema_initialized || fabsf(x - sp_gate_x_raw) > 2.0f) {
    // First detection or large jump (new gate): accept directly
    sp_gate_x_raw = x;
    sp_gate_y_raw = y;
    sp_gate_z_raw = z;
    sp_gate_ema_initialized = true;
  } else {
    // Smooth with EMA to reduce jitter
    sp_gate_x_raw = sp_gate_ema_alpha * x + (1.f - sp_gate_ema_alpha) * sp_gate_x_raw;
    sp_gate_y_raw = sp_gate_ema_alpha * y + (1.f - sp_gate_ema_alpha) * sp_gate_y_raw;
    sp_gate_z_raw = sp_gate_ema_alpha * z + (1.f - sp_gate_ema_alpha) * sp_gate_z_raw;
  }
  sp_gate_heading_at_detection = stateGetNedToBodyEulers_f()->psi;
  sp_last_detection_s = now;
}

static float clampf(float value, float min_value, float max_value)
{
  if (value < min_value) {
    return min_value;
  }
  if (value > max_value) {
    return max_value;
  }
  return value;
}

static float gate_distance_abs(void)
{
  return fmaxf(fabsf(sp_gate_x), 0.1f);
}

static void increase_nav_heading(float increment_degrees)
{
  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(increment_degrees);
  FLOAT_ANGLE_NORMALIZE(new_heading);
  nav.heading = new_heading;
}

static void move_waypoint_forward_lateral_alt(uint8_t waypoint, float forward_m, float lateral_m, float target_alt)
{
  struct EnuCoor_i new_coor;
  float heading = stateGetNedToBodyEulers_f()->psi;

  float dx = sinf(heading) * forward_m + cosf(heading) * lateral_m;
  float dy = cosf(heading) * forward_m - sinf(heading) * lateral_m;

  new_coor.x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(dx);
  new_coor.y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(dy);
  new_coor.z = POS_BFP_OF_REAL(clampf(target_alt, spg_search_min_altitude, spg_search_max_altitude));

  waypoint_move_enu_i(waypoint, &new_coor);
}

static void move_waypoint_to_absolute_alt(uint8_t waypoint, float x, float y, float target_alt)
{
  struct EnuCoor_i new_coor;
  new_coor.x = POS_BFP_OF_REAL(x);
  new_coor.y = POS_BFP_OF_REAL(y);
  new_coor.z = POS_BFP_OF_REAL(clampf(target_alt, spg_search_min_altitude, spg_search_max_altitude));
  waypoint_move_enu_i(waypoint, &new_coor);
}

static void choose_random_turn_direction(void)
{
  if (rand() % 2 == 0) {
    sp_turn_direction = 1.f;
  } else {
    sp_turn_direction = -1.f;
  }
}

static void start_search_turn(float turn_deg)
{
  sp_search_turn_remaining_deg = fmaxf(turn_deg, 0.f);
  sp_search_step_remaining_m = 0.f;
}

static void hold_current_position(float target_alt)
{
  float x = stateGetPositionEnu_f()->x;
  float y = stateGetPositionEnu_f()->y;
  move_waypoint_to_absolute_alt(WP_GOAL, x, y, target_alt);
  move_waypoint_to_absolute_alt(WP_TRAJECTORY, x, y, target_alt);
}

void square_passer_guided_init(void)
{
  srand(time(NULL));
  choose_random_turn_direction();

  AbiBindMsgRELATIVE_LOCALIZATION(SQUARE_PASSER_RELATIVE_LOCALIZATION_ID,
                                  &relative_localization_ev,
                                  relative_localization_cb);
}

void square_passer_guided_set_enabled(bool enabled)
{
  sp_enabled = enabled;
  if (!sp_enabled) {
    sp_state = SP_SEARCH;
    sp_pass_until_s = 0.f;
    sp_last_periodic_s = -1.f;
    sp_centered_since_s = -1.f;
    sp_track_seen_since_s = -1.f;
    sp_candidate_seen_since_s = -1.f;
    sp_search_altitude = 1.1f;
    sp_search_alt_dir = 1.f;
    sp_search_turn_remaining_deg = sp_search_scan_turn_deg;
    sp_search_step_remaining_m = 0.f;
    sp_search_anchor_valid = false;
    sp_gate_bearing_valid = false;
    sp_gate_ema_initialized = false;
  }
}

bool square_passer_guided_is_enabled(void)
{
  return sp_enabled;
}

void square_passer_guided_periodic(void)
{
  if (!sp_enabled || !autopilot_in_flight()) {
    sp_state = SP_SEARCH;
    sp_pass_until_s = 0.f;
    sp_last_periodic_s = -1.f;
    sp_centered_since_s = -1.f;
    sp_track_seen_since_s = -1.f;
    sp_candidate_seen_since_s = -1.f;
    sp_search_altitude = 1.1f;
    sp_search_alt_dir = 1.f;
    sp_search_turn_remaining_deg = sp_search_scan_turn_deg;
    sp_search_step_remaining_m = 0.f;
    sp_search_anchor_valid = false;
    sp_gate_bearing_valid = false;
    sp_gate_ema_initialized = false;
    sp_gate_velocity_initialized = false;
    sp_gate_vx_raw = 0.f;
    sp_gate_vy_raw = 0.f;
    sp_gate_vz_raw = 0.f;
    sp_prev_meas_s = -1.f;
    return;
  }

  float now = get_sys_time_float();
  float dt = 0.1f;
  if (sp_last_periodic_s > 0.f) {
    dt = now - sp_last_periodic_s;
    dt = clampf(dt, 0.01f, 0.5f);
  }
  sp_last_periodic_s = now;

  if (stateGetPositionEnu_f()->z < spg_min_guided_altitude) {
    sp_state = SP_SEARCH;
    sp_centered_since_s = -1.f;
    sp_track_seen_since_s = -1.f;
    sp_candidate_seen_since_s = -1.f;
    sp_search_altitude = clampf(stateGetPositionEnu_f()->z, spg_search_min_altitude, spg_search_max_altitude);
    sp_search_turn_remaining_deg = sp_search_scan_turn_deg;
    sp_search_step_remaining_m = 0.f;
    sp_search_anchor_valid = false;
    sp_gate_bearing_valid = false;
    sp_gate_ema_initialized = false;
    sp_gate_velocity_initialized = false;
    sp_gate_vx_raw = 0.f;
    sp_gate_vy_raw = 0.f;
    sp_gate_vz_raw = 0.f;
    sp_prev_meas_s = -1.f;
    return;
  }

  // Suppress detections during post-pass cooldown to avoid re-acquiring the same gate
  bool in_cooldown = (now - sp_pass_completed_at_s) < sp_post_pass_cooldown_s;
  bool gate_detected = !in_cooldown && (now - sp_last_detection_s) < spg_detection_timeout;
  bool gate_recent = !in_cooldown && (now - sp_last_detection_s) < (spg_detection_timeout + sp_track_lost_grace_s);

  // --- Latency compensation: rotate gate position from old body frame to current ---
  // The detection was computed relative to body heading at capture time.
  // Since the drone may have yawed since then, rotate the x/y coordinates.
  {
    float current_heading = stateGetNedToBodyEulers_f()->psi;
    float delta_psi = current_heading - sp_gate_heading_at_detection;
    // Normalize to [-pi, pi]
    while (delta_psi > M_PI)  { delta_psi -= 2.f * M_PI; }
    while (delta_psi < -M_PI) { delta_psi += 2.f * M_PI; }

    float cos_dp = cosf(delta_psi);
    float sin_dp = sinf(delta_psi);
    // Transform from old body frame to new body frame: P_new = R_delta^T * P_old
    sp_gate_x = sp_gate_x_raw * cos_dp + sp_gate_y_raw * sin_dp;
    sp_gate_y = -sp_gate_x_raw * sin_dp + sp_gate_y_raw * cos_dp;
    sp_gate_z = sp_gate_z_raw;  // altitude unaffected by yaw

    float gate_vx = sp_gate_vx_raw * cos_dp + sp_gate_vy_raw * sin_dp;
    float gate_vy = -sp_gate_vx_raw * sin_dp + sp_gate_vy_raw * cos_dp;
    float gate_vz = sp_gate_vz_raw;

    float meas_age = fmaxf(now - sp_last_detection_s, 0.f);
    float prediction_horizon = clampf(spg_camera_delay_comp_s + meas_age,
                                      0.f, spg_max_prediction_horizon_s);
    if (sp_gate_velocity_initialized && prediction_horizon > 0.f) {
      sp_gate_x += gate_vx * prediction_horizon;
      sp_gate_y += gate_vy * prediction_horizon;
      sp_gate_z += gate_vz * prediction_horizon;
    }
  }

  float gate_distance = gate_distance_abs();

  VERBOSE_PRINT("state=%d gate_detected=%d dist=%f y=%f z=%f dpsi=%.1f\n", sp_state, gate_detected, gate_distance, sp_gate_y,
                sp_gate_z, DegOfRad(stateGetNedToBodyEulers_f()->psi - sp_gate_heading_at_detection));

  switch (sp_state) {
    case SP_SEARCH:
      sp_centered_since_s = -1.f;

      // During post-pass cooldown, hold position and rotate to reacquire safely.
      if (in_cooldown) {
        float cooldown_alt = clampf(stateGetPositionEnu_f()->z, spg_search_min_altitude, spg_search_max_altitude);
        float heading_step_deg = DegOfRad(spg_search_heading_rate) * dt;
        increase_nav_heading(sp_turn_direction * heading_step_deg);
        hold_current_position(cooldown_alt);
        VERBOSE_PRINT("SEARCH (cooldown): rotate-hold, %.1fs remaining\n",
                      sp_post_pass_cooldown_s - (now - sp_pass_completed_at_s));
        break;
      }

      if (!sp_search_anchor_valid) {
        sp_search_anchor_x = stateGetPositionEnu_f()->x;
        sp_search_anchor_y = stateGetPositionEnu_f()->y;
        sp_search_altitude = clampf(stateGetPositionEnu_f()->z, spg_search_min_altitude, spg_search_max_altitude);
        sp_search_alt_dir = 1.f;
        start_search_turn(sp_search_scan_turn_deg);
        sp_search_anchor_valid = true;
      }

      sp_search_altitude = clampf(sp_search_altitude, spg_search_min_altitude, spg_search_max_altitude);

      // --- Reactive search: if we detect a gate, turn toward it immediately ---
      if (gate_detected) {
        // Compute bearing to gate in body frame: sp_gate_x is forward (negative = in front),
        // sp_gate_y is lateral. Turn toward the gate by yawing.
        float bearing_to_gate_deg = DegOfRad(atan2f(-sp_gate_y, -sp_gate_x));
        sp_last_gate_bearing_deg = bearing_to_gate_deg;
        sp_gate_bearing_valid = true;

        // Interrupt blind rotation — turn toward the gate
        if (fabsf(bearing_to_gate_deg) > 3.0f) {
          // Apply a fraction of the bearing correction per tick for smooth yaw
          float yaw_step = clampf(bearing_to_gate_deg, -DegOfRad(spg_search_heading_rate) * dt * 2.0f,
                                  DegOfRad(spg_search_heading_rate) * dt * 2.0f);
          increase_nav_heading(yaw_step);
          sp_search_turn_remaining_deg = 0.f;  // cancel blind 360
          VERBOSE_PRINT("SEARCH: gate detected at dist=%.1f, turning %.1f deg toward it\n",
                        gate_distance, bearing_to_gate_deg);
        }

        // Also creep forward toward the gate while turning
        move_waypoint_forward_lateral_alt(WP_GOAL, spg_search_forward_speed, 0.f, sp_search_altitude);
        move_waypoint_forward_lateral_alt(WP_TRAJECTORY, 1.5f * spg_search_forward_speed, 0.f, sp_search_altitude);

      } else if (sp_gate_bearing_valid) {
        // Gate was seen recently but lost — keep turning toward last known bearing
        float residual = sp_last_gate_bearing_deg;
        if (fabsf(residual) > 2.0f) {
          float yaw_step = clampf(residual, -DegOfRad(spg_search_heading_rate) * dt,
                                  DegOfRad(spg_search_heading_rate) * dt);
          increase_nav_heading(yaw_step);
          sp_last_gate_bearing_deg -= yaw_step;  // decay bearing memory
        } else {
          sp_gate_bearing_valid = false;  // bearing consumed
        }
        // Move slowly forward toward last known direction
        move_waypoint_forward_lateral_alt(WP_GOAL, 0.5f * spg_search_forward_speed, 0.f, sp_search_altitude);
        move_waypoint_forward_lateral_alt(WP_TRAJECTORY, spg_search_forward_speed, 0.f, sp_search_altitude);

      } else if (sp_search_turn_remaining_deg > 0.f) {
        // --- Blind rotation scan ---
        float heading_step_deg = DegOfRad(spg_search_heading_rate) * dt;
        float applied_deg = fminf(heading_step_deg, sp_search_turn_remaining_deg);
        increase_nav_heading(sp_turn_direction * applied_deg);
        sp_search_turn_remaining_deg -= applied_deg;
        float scan_forward = 0.35f * spg_search_forward_speed;
        move_waypoint_forward_lateral_alt(WP_GOAL, scan_forward, 0.f, sp_search_altitude);
        move_waypoint_forward_lateral_alt(WP_TRAJECTORY, 2.0f * scan_forward, 0.f, sp_search_altitude);
      } else {
        // --- Forward exploration step ---
        if (sp_search_step_remaining_m <= 0.f) {
          sp_search_step_remaining_m = sp_search_step_distance;
        }

        move_waypoint_forward_lateral_alt(WP_GOAL, spg_search_forward_speed, 0.f, sp_search_altitude);
        move_waypoint_forward_lateral_alt(WP_TRAJECTORY, 1.5f * spg_search_forward_speed, 0.f, sp_search_altitude);
        sp_search_step_remaining_m -= spg_search_forward_speed;

        if (!InsideCyberZoo(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY)) ||
            !InsideCyberZoo(stateGetPositionEnu_f()->x, stateGetPositionEnu_f()->y)) {
          sp_turn_direction = -sp_turn_direction;
          hold_current_position(sp_search_altitude);
          start_search_turn(sp_search_boundary_turn_deg);
        } else if (sp_search_step_remaining_m <= 0.f) {
          hold_current_position(sp_search_altitude);
          start_search_turn(sp_search_scan_turn_deg);
        }
      }

      // --- Transition to TRACK if gate within range and confirmed ---
      if (gate_detected && gate_distance < sp_max_track_distance) {
        if (sp_candidate_seen_since_s < 0.f) {
          sp_candidate_seen_since_s = now;
        }

        if ((now - sp_candidate_seen_since_s) > sp_lock_confirm_time) {
          sp_search_altitude = clampf(stateGetPositionEnu_f()->z, spg_search_min_altitude, spg_search_max_altitude);
          sp_state = SP_TRACK;
          sp_track_seen_since_s = now;
          sp_candidate_seen_since_s = -1.f;
          sp_gate_bearing_valid = false;
          sp_search_turn_remaining_deg = sp_search_scan_turn_deg;
          sp_search_step_remaining_m = 0.f;
          sp_search_anchor_valid = false;
          VERBOSE_PRINT("SEARCH -> TRACK: dist=%.2f\n", gate_distance);
        }
      } else {
        sp_candidate_seen_since_s = -1.f;
      }
      break;

    case SP_TRACK:
      if ((!gate_detected && !gate_recent) || gate_distance > (1.4f * sp_max_track_distance)) {
        VERBOSE_PRINT("TRACK -> SEARCH: lost gate (detected=%d, dist=%.2f)\n", gate_detected, gate_distance);
        // Remember where the gate was so search can look there
        sp_last_gate_bearing_deg = DegOfRad(atan2f(-sp_gate_y, -sp_gate_x));
        sp_gate_bearing_valid = true;
        sp_state = SP_SEARCH;
        sp_centered_since_s = -1.f;
        sp_track_seen_since_s = -1.f;
        sp_candidate_seen_since_s = -1.f;
        sp_search_turn_remaining_deg = sp_search_scan_turn_deg;
        sp_search_step_remaining_m = 0.f;
        sp_search_anchor_valid = false;
        break;
      }

      {
        if (sp_track_seen_since_s < 0.f) {
          sp_track_seen_since_s = now;
        }
        float seen_time = now - sp_track_seen_since_s;

        // --- Visual servoing: steer toward gate center ---
        float lateral_ratio = sp_gate_y / gate_distance;
        float vertical_ratio = sp_gate_z / gate_distance;
        float vertical_error = vertical_ratio - spg_vertical_target_ratio;
        bool lateral_centered = fabsf(lateral_ratio) < spg_gate_center_tolerance;
        bool vertical_centered = fabsf(vertical_error) < spg_gate_vertical_tolerance;
        bool vertical_soft = fabsf(vertical_error) < fmaxf(4.0f * spg_gate_vertical_tolerance, 0.65f);
        bool centered_now = lateral_centered && vertical_centered;
        bool nearly_centered = fabsf(lateral_ratio) < (2.5f * spg_gate_center_tolerance) &&
             fabsf(vertical_error) < fmaxf(2.5f * spg_gate_vertical_tolerance, 0.75f);

        if (centered_now) {
          if (sp_centered_since_s < 0.f) {
            sp_centered_since_s = now;
          }
        } else {
          sp_centered_since_s = -1.f;
        }

        // Yaw toward the gate for faster alignment (proportional navigation)
        if (fabsf(lateral_ratio) > 0.05f) {
          float yaw_correction = DegOfRad(atan2f(-sp_gate_y, -sp_gate_x));
          float yaw_step = clampf(yaw_correction * 0.3f, -8.0f, 8.0f);
          increase_nav_heading(yaw_step);
        }

        float lateral_cmd = clampf(-spg_lateral_gain * lateral_ratio, -spg_max_lateral_speed, spg_max_lateral_speed);
        // NB: drone_position.z is NED (z-down) while target_alt is ENU (z-up), so sign must be inverted.
        float vertical_cmd = clampf(-spg_vertical_gain * vertical_error, -spg_max_vertical_step, spg_max_vertical_step);

        // Distance-adaptive forward speed: faster when far and centered, slower when close for precision
        float distance_scale = clampf(gate_distance / 3.0f, 0.30f, 1.0f);
        float forward_cmd = 0.f;
        if (centered_now) {
          forward_cmd = spg_center_forward_speed * distance_scale;
        } else if (lateral_centered && vertical_soft) {
          forward_cmd = 0.70f * spg_center_forward_speed * distance_scale;
        } else if (nearly_centered) {
          forward_cmd = 0.50f * spg_center_forward_speed * distance_scale;
        } else if (seen_time > 0.20f) {
          // Only creep slowly — prioritize centering before approaching
          forward_cmd = 0.25f * spg_center_forward_speed * distance_scale;
        }

        float target_alt = clampf(stateGetPositionEnu_f()->z + vertical_cmd, spg_search_min_altitude, spg_search_max_altitude);

        move_waypoint_forward_lateral_alt(WP_GOAL, forward_cmd, lateral_cmd, target_alt);
        move_waypoint_forward_lateral_alt(WP_TRAJECTORY, 1.5f * forward_cmd, lateral_cmd, target_alt);

        VERBOSE_PRINT("TRACK: dist=%.2f lat=%.2f vert_err=%.2f fwd=%.2f centered=%d\n",
                gate_distance, lateral_ratio, vertical_error, forward_cmd, centered_now);

        // --- Commit to pass-through only when centered for long enough (or very close and still centered) ---
        bool commit_centered = gate_distance > 0.f &&
                               gate_distance < spg_gate_pass_distance &&
                               sp_centered_since_s > 0.f &&
                               (now - sp_centered_since_s) > spg_center_confirm_time;
        bool commit_close_centered = gate_distance > 0.f &&
                                     gate_distance < 1.05f &&
                                     fabsf(lateral_ratio) < (1.5f * spg_gate_center_tolerance) &&
                                     vertical_soft &&
                                     seen_time > 0.30f;
        bool commit_lateral_aligned = gate_distance > 0.f &&
                                      gate_distance < (spg_gate_pass_distance + 0.25f) &&
                                      lateral_centered &&
                                      vertical_soft &&
                                      seen_time > 0.60f;

        if (commit_centered || commit_close_centered || commit_lateral_aligned) {
          VERBOSE_PRINT("TRACK -> PASS_THROUGH: dist=%.2f (centered=%d close=%d lateral=%d)\n",
                        gate_distance, commit_centered, commit_close_centered, commit_lateral_aligned);
          sp_pass_until_s = now + spg_pass_through_time;
          sp_state = SP_PASS_THROUGH;
          sp_centered_since_s = -1.f;
          sp_track_seen_since_s = -1.f;
        }
      }

      if (!InsideCyberZoo(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY)) ||
          !InsideCyberZoo(stateGetPositionEnu_f()->x, stateGetPositionEnu_f()->y)) {
        choose_random_turn_direction();
        increase_nav_heading(sp_turn_direction * 10.f);
        sp_state = SP_SEARCH;
        sp_centered_since_s = -1.f;
        sp_track_seen_since_s = -1.f;
        sp_candidate_seen_since_s = -1.f;
        sp_search_turn_remaining_deg = sp_search_boundary_turn_deg;
        sp_search_step_remaining_m = 0.f;
        sp_search_anchor_valid = false;
      }
      break;

    case SP_PASS_THROUGH:
      {
        float pass_elapsed = spg_pass_through_time - fmaxf(sp_pass_until_s - now, 0.f);
        float lateral_cmd = 0.f;
        float vertical_cmd = 0.f;
        float forward_cmd = spg_forward_speed;
        if (gate_detected) {
          // Keep correcting laterally while flying through
          float lateral_ratio = sp_gate_y / gate_distance;
          float vertical_ratio = sp_gate_z / gate_distance;
          float vertical_error = vertical_ratio - spg_vertical_target_ratio;
          lateral_cmd = clampf(-spg_lateral_gain * lateral_ratio, -0.22f, 0.22f);
          // NB: NED z-down vs ENU z-up → invert sign
          vertical_cmd = clampf(-spg_vertical_gain * vertical_error, -0.18f, 0.18f);
          // Also yaw toward gate during pass-through for better alignment
          if (fabsf(lateral_ratio) > 0.05f) {
            float yaw_step = clampf(DegOfRad(atan2f(-sp_gate_y, -sp_gate_x)) * 0.25f, -5.0f, 5.0f);
            increase_nav_heading(yaw_step);
          }
        } else if (pass_elapsed > 0.35f) {
          // Once committed and gate is gone, reduce blind-forward motion.
          forward_cmd = 0.65f * spg_forward_speed;
        }
        float target_alt = clampf(stateGetPositionEnu_f()->z + vertical_cmd, spg_search_min_altitude, spg_search_max_altitude);
        move_waypoint_forward_lateral_alt(WP_GOAL, forward_cmd, lateral_cmd, target_alt);
        move_waypoint_forward_lateral_alt(WP_TRAJECTORY, 1.5f * forward_cmd, lateral_cmd, target_alt);
        VERBOSE_PRINT("PASS_THROUGH: fwd=%.2f lat=%.2f elapsed=%.1fs remaining=%.1fs\n",
                      forward_cmd, lateral_cmd, pass_elapsed, sp_pass_until_s - now);

        bool pass_timeout = now > sp_pass_until_s;
        bool pass_gate_lost = !gate_detected && pass_elapsed > sp_pass_exit_no_gate_s;

        if (pass_timeout || pass_gate_lost) {
          choose_random_turn_direction();
          sp_state = SP_SEARCH;
          sp_pass_completed_at_s = now;
          sp_track_seen_since_s = -1.f;
          sp_candidate_seen_since_s = -1.f;
          sp_search_turn_remaining_deg = sp_search_scan_turn_deg;
          sp_search_step_remaining_m = 0.f;
          sp_search_anchor_valid = false;
          sp_gate_ema_initialized = false;
          VERBOSE_PRINT("PASS_THROUGH complete (%s), cooldown %.1fs\n",
                        pass_timeout ? "timeout" : "gate_lost", sp_post_pass_cooldown_s);
        }
      }

      if (!InsideCyberZoo(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY)) ||
          !InsideCyberZoo(stateGetPositionEnu_f()->x, stateGetPositionEnu_f()->y)) {
        choose_random_turn_direction();
        increase_nav_heading(sp_turn_direction * DegOfRad(spg_search_heading_rate * 3.0f));
        sp_state = SP_SEARCH;
        sp_pass_until_s = 0.f;
        sp_track_seen_since_s = -1.f;
        sp_candidate_seen_since_s = -1.f;
        sp_search_turn_remaining_deg = sp_search_boundary_turn_deg;
        sp_search_step_remaining_m = 0.f;
        sp_search_anchor_valid = false;
        move_waypoint_to_absolute_alt(WP_GOAL,
                                      stateGetPositionEnu_f()->x,
                                      stateGetPositionEnu_f()->y,
                                      stateGetPositionEnu_f()->z);
      }
      break;

    default:
      break;
  }
}
