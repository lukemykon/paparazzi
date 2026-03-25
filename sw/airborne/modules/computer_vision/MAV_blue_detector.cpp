/**
 * @file MAV_blue_detector.cpp
 * @brief YUV-based blue pixel detector, modeled after MAV_plant_avoider.
 *
 * Logic:
 *  - Detect blue pixels directly in YUV422.
 *  - Optionally restrict to half of the frame.
 *  - Split detections into three zones (left/straight/right).
 *  - Optionally draw a mask over matched pixels in the camera stream.
 *  - Optionally drive plant-avoider-style turn/straight body-velocity commands.
 */

#include "MAV_blue_detector.h"

extern "C" {
#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "firmwares/rotorcraft/navigation.h"
#include "state.h"
}

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>

#ifndef BLUE_DETECTOR_CAMERA
#define BLUE_DETECTOR_CAMERA front_camera
#endif

#ifndef BLUE_DETECTOR_FPS
#define BLUE_DETECTOR_FPS 4
#endif

#ifndef BLUE_DETECTOR_ENABLE_VISION_CALLBACK
#define BLUE_DETECTOR_ENABLE_VISION_CALLBACK 1
#endif

#ifndef BLUE_DETECTOR_ENABLE_STANDALONE_CONTROL
#ifdef BLUE_DETECTOR_ENABLE_ROAM_CONTROL
#define BLUE_DETECTOR_ENABLE_STANDALONE_CONTROL BLUE_DETECTOR_ENABLE_ROAM_CONTROL
#else
#define BLUE_DETECTOR_ENABLE_STANDALONE_CONTROL 0
#endif
#endif

/* In NAV mode, route guidance can overwrite speed setpoints.
 * Enable this to let detector commands drive horizontal guidance in simulation. */
#ifndef BLUE_DETECTOR_NAV_MODE_CONTROL
#define BLUE_DETECTOR_NAV_MODE_CONTROL 0
#endif

#ifndef BLUE_DETECTOR_SHOW_MASK
#define BLUE_DETECTOR_SHOW_MASK 1
#endif

#ifndef BLUE_DETECTOR_ROTATED_CAMERA_TOP_HALF
#define BLUE_DETECTOR_ROTATED_CAMERA_TOP_HALF 1
#endif

#ifndef BLUE_DETECTOR_USE_HALF_FOV
#define BLUE_DETECTOR_USE_HALF_FOV 1
#endif

#ifndef BLUE_DETECTOR_GROUND_ALT_M
#define BLUE_DETECTOR_GROUND_ALT_M 0.50f
#endif

#ifndef BLUE_DETECTOR_FORWARD_SPEED
#define BLUE_DETECTOR_FORWARD_SPEED 0.3f
#endif

#ifndef BLUE_DETECTOR_TURN_SPEED
#define BLUE_DETECTOR_TURN_SPEED 0.3f
#endif

#ifndef BLUE_DETECTOR_STRAIGHT_SAFE_PCT
#define BLUE_DETECTOR_STRAIGHT_SAFE_PCT 40.0f
#endif

/* YUV mask color for matched blue pixels. */
#ifndef BLUE_DETECTOR_MASK_Y
#define BLUE_DETECTOR_MASK_Y 110U
#endif
#ifndef BLUE_DETECTOR_MASK_U
#define BLUE_DETECTOR_MASK_U 220U
#endif
#ifndef BLUE_DETECTOR_MASK_V
#define BLUE_DETECTOR_MASK_V 80U
#endif

/* Default blue YUV profile (tune if needed). */
#ifndef BLUE_DETECTOR_Y_CENTER
#define BLUE_DETECTOR_Y_CENTER 3
#endif
#ifndef BLUE_DETECTOR_U_CENTER
#define BLUE_DETECTOR_U_CENTER 81
#endif
#ifndef BLUE_DETECTOR_V_CENTER
#define BLUE_DETECTOR_V_CENTER -50
#endif

#ifndef BLUE_DETECTOR_Y_TOL
#define BLUE_DETECTOR_Y_TOL 30
#endif
#ifndef BLUE_DETECTOR_U_TOL
#define BLUE_DETECTOR_U_TOL 30
#endif
#ifndef BLUE_DETECTOR_V_TOL
#define BLUE_DETECTOR_V_TOL 30
#endif

#ifndef BLUE_DETECTOR_SIGNED_CENTERS
#define BLUE_DETECTOR_SIGNED_CENTERS 0
#endif

#if BLUE_DETECTOR_SIGNED_CENTERS
#define BLUE_DETECTOR_U_CENTER_SIGNED BLUE_DETECTOR_U_CENTER
#define BLUE_DETECTOR_V_CENTER_SIGNED BLUE_DETECTOR_V_CENTER
#else
#define BLUE_DETECTOR_U_CENTER_SIGNED (BLUE_DETECTOR_U_CENTER - 128)
#define BLUE_DETECTOR_V_CENTER_SIGNED (BLUE_DETECTOR_V_CENTER - 128)
#endif

#ifndef BLUE_DETECTOR_Y_MIN
#define BLUE_DETECTOR_Y_MIN (BLUE_DETECTOR_Y_CENTER - BLUE_DETECTOR_Y_TOL)
#endif
#ifndef BLUE_DETECTOR_Y_MAX
#define BLUE_DETECTOR_Y_MAX (BLUE_DETECTOR_Y_CENTER + BLUE_DETECTOR_Y_TOL)
#endif
#ifndef BLUE_DETECTOR_U_MIN
#define BLUE_DETECTOR_U_MIN (BLUE_DETECTOR_U_CENTER_SIGNED - BLUE_DETECTOR_U_TOL)
#endif
#ifndef BLUE_DETECTOR_U_MAX
#define BLUE_DETECTOR_U_MAX (BLUE_DETECTOR_U_CENTER_SIGNED + BLUE_DETECTOR_U_TOL)
#endif
#ifndef BLUE_DETECTOR_V_MIN
#define BLUE_DETECTOR_V_MIN (BLUE_DETECTOR_V_CENTER_SIGNED - BLUE_DETECTOR_V_TOL)
#endif
#ifndef BLUE_DETECTOR_V_MAX
#define BLUE_DETECTOR_V_MAX (BLUE_DETECTOR_V_CENTER_SIGNED + BLUE_DETECTOR_V_TOL)
#endif

struct bd_zone_scores_t {
  uint32_t left;
  uint32_t straight;
  uint32_t right;
  uint32_t total;
};

float bd_load_left = 0.0f;
float bd_load_straight = 0.0f;
float bd_load_right = 0.0f;
float bd_forward_speed = BLUE_DETECTOR_FORWARD_SPEED;
float bd_turn_speed = BLUE_DETECTOR_TURN_SPEED;
int8_t bd_last_direction = 0;
float bd_weight_left = 0.0f;
float bd_weight_straight = 1.0f;
float bd_weight_right = 0.0f;

static struct bd_zone_scores_t g_scores;
static pthread_mutex_t g_mutex;

static inline bool is_blue_yuv(uint8_t y, uint8_t u, uint8_t v)
{
  const int16_t y_signed = (int16_t)y;
  const int16_t u_signed = (int16_t)u - 128;
  const int16_t v_signed = (int16_t)v - 128;
  return (y_signed >= BLUE_DETECTOR_Y_MIN && y_signed <= BLUE_DETECTOR_Y_MAX &&
          u_signed >= BLUE_DETECTOR_U_MIN && u_signed <= BLUE_DETECTOR_U_MAX &&
          v_signed >= BLUE_DETECTOR_V_MIN && v_signed <= BLUE_DETECTOR_V_MAX);
}

#if BLUE_DETECTOR_ENABLE_STANDALONE_CONTROL
static bool is_drone_near_ground(void)
{
  return stateGetPositionEnu_f()->z <= BLUE_DETECTOR_GROUND_ALT_M;
}
#endif

static void detect_blue_top_half(struct image_t *img, bool draw_mask, struct bd_zone_scores_t *out)
{
  memset(out, 0, sizeof(*out));

  const uint16_t w = img->w;
  const uint16_t h = img->h;
  const uint16_t one_third_y = h / 3;
#if !BLUE_DETECTOR_ROTATED_CAMERA_TOP_HALF
  const uint16_t one_third_x = w / 3;
#endif

  uint8_t *buf = (uint8_t *)img->buf;

  for (uint16_t y = 0; y < h; y++) {
    const uint32_t row_base = (uint32_t)y * 2U * (uint32_t)w;

    for (uint16_t x = 0; x < w; x++) {
#if BLUE_DETECTOR_ROTATED_CAMERA_TOP_HALF
      if (BLUE_DETECTOR_USE_HALF_FOV && x < (w / 2U)) {
        continue;
      }
#else
      if (BLUE_DETECTOR_USE_HALF_FOV && y >= (h / 2U)) {
        continue;
      }
#endif

      uint8_t yp;
      uint8_t up;
      uint8_t vp;

      if ((x & 1U) == 0U) {
        const uint32_t base = row_base + (uint32_t)(2U * x);
        up = buf[base];
        yp = buf[base + 1U];
        vp = buf[base + 2U];
      } else {
        const uint32_t base = row_base + (uint32_t)(2U * x);
        up = buf[base - 2U];
        yp = buf[base + 1U];
        vp = buf[base];
      }

      if (!is_blue_yuv(yp, up, vp)) {
        continue;
      }

#if BLUE_DETECTOR_SHOW_MASK
      if (draw_mask) {
        const uint16_t x_pair = (uint16_t)(x & (uint16_t)~1U);
        const uint32_t pair_base = row_base + (uint32_t)(2U * x_pair);
        const uint32_t row_end = row_base + (uint32_t)(2U * w);
        if ((pair_base + 3U) < row_end) {
          buf[pair_base] = BLUE_DETECTOR_MASK_U;
          buf[pair_base + 2U] = BLUE_DETECTOR_MASK_V;
          if ((x & 1U) == 0U) {
            buf[pair_base + 1U] = BLUE_DETECTOR_MASK_Y;
          } else {
            buf[pair_base + 3U] = BLUE_DETECTOR_MASK_Y;
          }
        }
      }
#endif

#if BLUE_DETECTOR_ROTATED_CAMERA_TOP_HALF
      if (y < one_third_y) {
        out->left++;
      } else if (y < (2U * one_third_y)) {
        out->straight++;
      } else {
        out->right++;
      }
#else
      if (x < one_third_x) {
        out->left++;
      } else if (x < (2U * one_third_x)) {
        out->straight++;
      } else {
        out->right++;
      }
#endif
      out->total++;
    }
  }
}

extern "C" void blue_detector_detect_losses(struct image_t *img, bool draw_mask,
                                             uint32_t *left, uint32_t *straight, uint32_t *right)
{
  if (!img || !img->buf || img->type != IMAGE_YUV422) {
    if (left) {
      *left = 0U;
    }
    if (straight) {
      *straight = 0U;
    }
    if (right) {
      *right = 0U;
    }
    return;
  }

  struct bd_zone_scores_t s;
  detect_blue_top_half(img, draw_mask, &s);

  if (left) {
    *left = s.left;
  }
  if (straight) {
    *straight = s.straight;
  }
  if (right) {
    *right = s.right;
  }
}

static struct image_t *blue_detector_func(struct image_t *img, uint8_t camera_id)
{
  (void)camera_id;

  if (!img || !img->buf || img->type != IMAGE_YUV422) {
    return img;
  }

  struct bd_zone_scores_t s;
  detect_blue_top_half(img, true, &s);

  pthread_mutex_lock(&g_mutex);
  g_scores = s;
  pthread_mutex_unlock(&g_mutex);

  return img;
}

extern "C" void blue_detector_init(void)
{
  memset(&g_scores, 0, sizeof(g_scores));
  pthread_mutex_init(&g_mutex, NULL);

  bd_forward_speed = BLUE_DETECTOR_FORWARD_SPEED;
  bd_turn_speed = BLUE_DETECTOR_TURN_SPEED;

#if BLUE_DETECTOR_ENABLE_VISION_CALLBACK
  cv_add_to_device(&BLUE_DETECTOR_CAMERA, blue_detector_func, BLUE_DETECTOR_FPS, 0);
#endif
}

extern "C" void blue_detector_periodic(void)
{
#if !BLUE_DETECTOR_ENABLE_STANDALONE_CONTROL
  return;
#else
  const bool grounded = is_drone_near_ground();
  const bool control_mode_active = (guidance_h.mode == GUIDANCE_H_MODE_GUIDED) ||
                                   (guidance_h.mode == GUIDANCE_H_MODE_NAV);

  struct bd_zone_scores_t s;
  pthread_mutex_lock(&g_mutex);
  s = g_scores;
  pthread_mutex_unlock(&g_mutex);

  if (s.total > 0U && !grounded) {
    bd_load_left = (100.0f * (float)s.left) / (float)s.total;
    bd_load_straight = (100.0f * (float)s.straight) / (float)s.total;
    bd_load_right = (100.0f * (float)s.right) / (float)s.total;
  } else {
    bd_load_left = 0.0f;
    bd_load_straight = 0.0f;
    bd_load_right = 0.0f;
  }

  int8_t direction = 0;
  if (!grounded && s.total > 0U) {
    const bool middle_is_safe = (bd_load_straight < BLUE_DETECTOR_STRAIGHT_SAFE_PCT);
    const bool left_is_best = (bd_load_left < bd_load_straight) &&
                              (bd_load_left < bd_load_right);

    if (middle_is_safe) {
      direction = 0;
    } else if (left_is_best) {
      direction = -1;
    } else {
      direction = 1;
    }
  }

  bd_weight_left = (direction < 0) ? 1.0f : 0.0f;
  bd_weight_straight = (direction == 0) ? 1.0f : 0.0f;
  bd_weight_right = (direction > 0) ? 1.0f : 0.0f;

  bd_last_direction = direction;

  float vx = bd_forward_speed;
  float vy = 0.0f;

  if (direction != 0) {
    vx = 0.0f;
  }

  if (direction < 0) {
    vy = -bd_turn_speed;
  } else if (direction > 0) {
    vy = bd_turn_speed;
  }

  if (control_mode_active && !grounded) {
#if BLUE_DETECTOR_NAV_MODE_CONTROL
    if (guidance_h.mode == GUIDANCE_H_MODE_NAV) {
      nav.horizontal_mode = NAV_HORIZONTAL_MODE_GUIDED;
      nav.setpoint_mode = NAV_SETPOINT_MODE_SPEED;
    }
#endif
    guidance_h_set_body_vel(vx, vy);
  }
#endif
}
