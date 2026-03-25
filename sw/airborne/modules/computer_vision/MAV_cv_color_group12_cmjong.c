#include "MAV_cv_color_group12_cmjong.h"
#include "generated/airframe.h"
#include <stdint.h>

#ifndef ORANGE_OBJECT_DETECTOR_LUM_MIN
#ifdef COLOR_OBJECT_DETECTOR_LUM_MIN1
#define ORANGE_OBJECT_DETECTOR_LUM_MIN COLOR_OBJECT_DETECTOR_LUM_MIN1
#define ORANGE_OBJECT_DETECTOR_LUM_MAX COLOR_OBJECT_DETECTOR_LUM_MAX1
#define ORANGE_OBJECT_DETECTOR_CB_MIN COLOR_OBJECT_DETECTOR_CB_MIN1
#define ORANGE_OBJECT_DETECTOR_CB_MAX COLOR_OBJECT_DETECTOR_CB_MAX1
#define ORANGE_OBJECT_DETECTOR_CR_MIN COLOR_OBJECT_DETECTOR_CR_MIN1
#define ORANGE_OBJECT_DETECTOR_CR_MAX COLOR_OBJECT_DETECTOR_CR_MAX1
#else
#define ORANGE_OBJECT_DETECTOR_LUM_MIN 30
#define ORANGE_OBJECT_DETECTOR_LUM_MAX 190
#define ORANGE_OBJECT_DETECTOR_CB_MIN 70
#define ORANGE_OBJECT_DETECTOR_CB_MAX 130
#define ORANGE_OBJECT_DETECTOR_CR_MIN 150
#define ORANGE_OBJECT_DETECTOR_CR_MAX 190
#endif
#endif

#ifndef GREEN_OBJECT_DETECTOR_LUM_MIN
#define GREEN_OBJECT_DETECTOR_LUM_MIN 18
#define GREEN_OBJECT_DETECTOR_LUM_MAX 170
#define GREEN_OBJECT_DETECTOR_CB_MIN 70
#define GREEN_OBJECT_DETECTOR_CB_MAX 145
#define GREEN_OBJECT_DETECTOR_CR_MIN 20
#define GREEN_OBJECT_DETECTOR_CR_MAX 120
#endif

#ifndef BLUE_OBJECT_DETECTOR_LUM_MIN
#define BLUE_OBJECT_DETECTOR_LUM_MIN 20
#define BLUE_OBJECT_DETECTOR_LUM_MAX 200
#define BLUE_OBJECT_DETECTOR_CB_MIN 150
#define BLUE_OBJECT_DETECTOR_CB_MAX 255
#define BLUE_OBJECT_DETECTOR_CR_MIN 0
#define BLUE_OBJECT_DETECTOR_CR_MAX 120
#endif

#ifndef BLUE_OBJECT_DETECTOR_SIGNED_CENTER
#define BLUE_OBJECT_DETECTOR_SIGNED_CENTER 0
#endif

#ifndef BLUE_OBJECT_DETECTOR_Y_CENTER
#define BLUE_OBJECT_DETECTOR_Y_CENTER ((BLUE_OBJECT_DETECTOR_LUM_MIN + BLUE_OBJECT_DETECTOR_LUM_MAX) / 2)
#endif

#ifndef BLUE_OBJECT_DETECTOR_Y_TOL
#define BLUE_OBJECT_DETECTOR_Y_TOL ((BLUE_OBJECT_DETECTOR_LUM_MAX - BLUE_OBJECT_DETECTOR_LUM_MIN) / 2)
#endif

#if BLUE_OBJECT_DETECTOR_SIGNED_CENTER
#ifndef BLUE_OBJECT_DETECTOR_U_CENTER
#define BLUE_OBJECT_DETECTOR_U_CENTER 81
#endif
#ifndef BLUE_OBJECT_DETECTOR_V_CENTER
#define BLUE_OBJECT_DETECTOR_V_CENTER -50
#endif
#else
#ifndef BLUE_OBJECT_DETECTOR_U_CENTER
#define BLUE_OBJECT_DETECTOR_U_CENTER (((BLUE_OBJECT_DETECTOR_CB_MIN + BLUE_OBJECT_DETECTOR_CB_MAX) / 2) - 128)
#endif
#ifndef BLUE_OBJECT_DETECTOR_V_CENTER
#define BLUE_OBJECT_DETECTOR_V_CENTER (((BLUE_OBJECT_DETECTOR_CR_MIN + BLUE_OBJECT_DETECTOR_CR_MAX) / 2) - 128)
#endif
#endif

#ifndef BLUE_OBJECT_DETECTOR_U_TOL
#define BLUE_OBJECT_DETECTOR_U_TOL ((BLUE_OBJECT_DETECTOR_CB_MAX - BLUE_OBJECT_DETECTOR_CB_MIN) / 2)
#endif

#ifndef BLUE_OBJECT_DETECTOR_V_TOL
#define BLUE_OBJECT_DETECTOR_V_TOL ((BLUE_OBJECT_DETECTOR_CR_MAX - BLUE_OBJECT_DETECTOR_CR_MIN) / 2)
#endif

#ifndef MAV_CV_WEIGHT_ORANGE
#define MAV_CV_WEIGHT_ORANGE 1.0f
#endif

#ifndef MAV_CV_WEIGHT_GREEN
#define MAV_CV_WEIGHT_GREEN 1.0f
#endif

#ifndef MAV_CV_WEIGHT_BLUE
#define MAV_CV_WEIGHT_BLUE 1.0f
#endif

#ifndef MAV_CV_BLUE_MASK_Y
#define MAV_CV_BLUE_MASK_Y 100U
#endif

#ifndef MAV_CV_BLUE_MASK_U
#define MAV_CV_BLUE_MASK_U 220U
#endif

#ifndef MAV_CV_BLUE_MASK_V
#define MAV_CV_BLUE_MASK_V 60U
#endif

#ifndef MAV_CV_FORCE_BLUE_MASK
#define MAV_CV_FORCE_BLUE_MASK 1
#endif

#ifndef COLOR_OBJECT_DETECTOR_DRAW_GUIDES
#define COLOR_OBJECT_DETECTOR_DRAW_GUIDES 0
#endif

/*
YUV422 memory layout (4 bytes per 2 pixels):
   [ U | Y1 | V | Y2 ]
     0    1   2    3
 Even pixel x reads:  U @ 2x,   Y1 @ 2x+1, V @ 2x+2
 Odd  pixel x reads:  U @ 2x-2, V @ 2x,    Y2 @ 2x+1

 CHANGED: regions are now split on Y axis (rows) instead of X axis (columns)
 because the Bebop front camera is mounted rotated 90 degrees CW.
 "left" = top of physical image, "right" = bottom of physical image.

Region layout (overlap = img->h / 12, half_overlap = img->h / 24):
   |<------- left ------->|
   0              h/3 + overlap
           |<-- middle -->|
           h/3 - half_overlap    2h/3 + half_overlap
                      |<------- right ------->|
                      2h/3 - overlap          h
*/


static void draw_horizontal_line(uint8_t *buffer, uint16_t img_w,
                                  uint16_t y_row, uint8_t y_val, uint8_t u_val, uint8_t v_val)
{
  for (uint16_t x = 0; x < img_w; x++) {
    uint16_t x_even = (x % 2 == 0) ? x : x - 1;
    uint32_t base = y_row * 2 * img_w + 2 * x_even;
    buffer[base]     = u_val;
    buffer[base + 1] = y_val;
    buffer[base + 2] = v_val;
    buffer[base + 3] = y_val;
  }
}

static inline uint16_t clamp_u16_from_float(float v)
{
  if (v <= 0.0f) {
    return 0U;
  }
  if (v >= 65535.0f) {
    return 65535U;
  }
  return (uint16_t)(v + 0.5f);
}

static inline uint16_t clamp_i16_range_u16(uint16_t v)
{
  return (v > 32767U) ? 32767U : v;
}

static PixelCount color_detection_signed_uv(struct image_t *img,
                                            int16_t y_center,
                                            int16_t y_tol,
                                            int16_t u_center_signed,
                                            int16_t u_tol,
                                            int16_t v_center_signed,
                                            int16_t v_tol,
                                            bool draw)
{
  uint8_t *buffer = img->buf;

  uint16_t overlap      = img->h / 12;
  uint16_t half_overlap = img->h / 24;
  uint16_t one_third    = img->h / 3;

  uint16_t left_end     = one_third + overlap;
  uint16_t middle_start = one_third - half_overlap;
  uint16_t middle_end   = 2 * one_third + half_overlap;
  uint16_t right_start  = 2 * one_third - overlap;

  const int16_t y_min = y_center - y_tol;
  const int16_t y_max = y_center + y_tol;
  const int16_t u_min = u_center_signed - u_tol;
  const int16_t u_max = u_center_signed + u_tol;
  const int16_t v_min = v_center_signed - v_tol;
  const int16_t v_max = v_center_signed + v_tol;

  PixelCount result = {0, 0, 0};

  for (uint16_t y = 0; y < img->h; y++) {
    for (uint16_t x = 0; x < img->w; x++) {
      uint8_t *yp, *up, *vp;
      if (x % 2 == 0) {
        up = &buffer[y * 2 * img->w + 2 * x];
        yp = &buffer[y * 2 * img->w + 2 * x + 1];
        vp = &buffer[y * 2 * img->w + 2 * x + 2];
      } else {
        up = &buffer[y * 2 * img->w + 2 * x - 2];
        vp = &buffer[y * 2 * img->w + 2 * x];
        yp = &buffer[y * 2 * img->w + 2 * x + 1];
      }

      const int16_t y_signed = (int16_t)(*yp);
      const int16_t u_signed = (int16_t)(*up) - 128;
      const int16_t v_signed = (int16_t)(*vp) - 128;

      if (y_signed >= y_min && y_signed <= y_max &&
          u_signed >= u_min && u_signed <= u_max &&
          v_signed >= v_min && v_signed <= v_max) {

        if (draw) {
          uint16_t x_even = (x % 2 == 0) ? x : (uint16_t)(x - 1U);
          uint32_t mask_base = y * 2U * img->w + 2U * x_even;
          buffer[mask_base]     = MAV_CV_BLUE_MASK_U;
          buffer[mask_base + 1] = MAV_CV_BLUE_MASK_Y;
          buffer[mask_base + 2] = MAV_CV_BLUE_MASK_V;
          buffer[mask_base + 3] = MAV_CV_BLUE_MASK_Y;
        }

        if (y < left_end)                        { result.left++;   }
        if (y >= middle_start && y < middle_end) { result.middle++; }
        if (y >= right_start)                    { result.right++;  }
      }
    }
  }

  return result;
}

PixelCount color_detection(struct image_t *img,
                             uint8_t lum_min, uint8_t lum_max,
                             uint8_t cb_min,  uint8_t cb_max,
                             uint8_t cr_min,  uint8_t cr_max,
                             bool draw)
{
  uint8_t *buffer = img->buf;

  uint16_t overlap      = img->h / 12;
  uint16_t half_overlap = img->h / 24;
  uint16_t one_third    = img->h / 3;

  uint16_t left_end     = one_third + overlap;
  uint16_t middle_start = one_third - half_overlap;
  uint16_t middle_end   = 2 * one_third + half_overlap;
  uint16_t right_start  = 2 * one_third - overlap;

  PixelCount result = {0, 0, 0};

  for (uint16_t y = 0; y < img->h; y++) {
    for (uint16_t x = 0; x < img->w; x++) {
      uint8_t *yp, *up, *vp;
      if (x % 2 == 0) {
        up = &buffer[y * 2 * img->w + 2 * x];
        yp = &buffer[y * 2 * img->w + 2 * x + 1];
        vp = &buffer[y * 2 * img->w + 2 * x + 2];
      } else {
        up = &buffer[y * 2 * img->w + 2 * x - 2];
        vp = &buffer[y * 2 * img->w + 2 * x];
        yp = &buffer[y * 2 * img->w + 2 * x + 1];
      }

      if ((*yp >= lum_min) && (*yp <= lum_max) &&
          (*up >= cb_min)  && (*up <= cb_max)  &&
          (*vp >= cr_min)  && (*vp <= cr_max)) {

        if (draw) {
          uint16_t x_even = (x % 2 == 0) ? x : (uint16_t)(x - 1U);
          uint32_t mask_base = y * 2U * img->w + 2U * x_even;
          buffer[mask_base]     = MAV_CV_BLUE_MASK_U;
          buffer[mask_base + 1] = MAV_CV_BLUE_MASK_Y;
          buffer[mask_base + 2] = MAV_CV_BLUE_MASK_V;
          buffer[mask_base + 3] = MAV_CV_BLUE_MASK_Y;
        }

        if (y < left_end)                        { result.left++;   }
        if (y >= middle_start && y < middle_end) { result.middle++; }
        if (y >= right_start)                    { result.right++;  }
      }
    }
  }

  if (draw && COLOR_OBJECT_DETECTOR_DRAW_GUIDES) {
    draw_horizontal_line(buffer, img->w, left_end,     255, 128, 128); // white
    draw_horizontal_line(buffer, img->w, middle_start, 255, 128, 128); // white
    draw_horizontal_line(buffer, img->w, middle_end,   255, 128, 128); // white
    draw_horizontal_line(buffer, img->w, right_start,  255, 128, 128); // white

    draw_horizontal_line(buffer, img->w, img->h / 6,       150, 90,  90);  // green  = left center
    draw_horizontal_line(buffer, img->w, img->h / 2,       150, 80,  180); // orange = middle center
    draw_horizontal_line(buffer, img->w, 5 * img->h / 6,   150, 90,  90);  // green  = right center
  }

  return result;
}

WeightedLoss compute_weighted_color_losses(struct image_t *img, bool draw_blue_mask)
{
  const PixelCount orange = color_detection(img,
                                            ORANGE_OBJECT_DETECTOR_LUM_MIN,
                                            ORANGE_OBJECT_DETECTOR_LUM_MAX,
                                            ORANGE_OBJECT_DETECTOR_CB_MIN,
                                            ORANGE_OBJECT_DETECTOR_CB_MAX,
                                            ORANGE_OBJECT_DETECTOR_CR_MIN,
                                            ORANGE_OBJECT_DETECTOR_CR_MAX,
                                            false);

  const PixelCount green = color_detection(img,
                                           GREEN_OBJECT_DETECTOR_LUM_MIN,
                                           GREEN_OBJECT_DETECTOR_LUM_MAX,
                                           GREEN_OBJECT_DETECTOR_CB_MIN,
                                           GREEN_OBJECT_DETECTOR_CB_MAX,
                                           GREEN_OBJECT_DETECTOR_CR_MIN,
                                           GREEN_OBJECT_DETECTOR_CR_MAX,
                                           false);

  const bool draw_blue = (draw_blue_mask || (MAV_CV_FORCE_BLUE_MASK != 0));

  const PixelCount blue = color_detection_signed_uv(img,
                                                    BLUE_OBJECT_DETECTOR_Y_CENTER,
                                                    BLUE_OBJECT_DETECTOR_Y_TOL,
                                                    BLUE_OBJECT_DETECTOR_U_CENTER,
                                                    BLUE_OBJECT_DETECTOR_U_TOL,
                                                    BLUE_OBJECT_DETECTOR_V_CENTER,
                                                    BLUE_OBJECT_DETECTOR_V_TOL,
                                                    draw_blue);

  WeightedLoss out;
  out.left = clamp_i16_range_u16(clamp_u16_from_float(MAV_CV_WEIGHT_ORANGE * (float)orange.left +
                                                      MAV_CV_WEIGHT_GREEN * (float)green.left +
                                                      MAV_CV_WEIGHT_BLUE * (float)blue.left));
  out.middle = clamp_i16_range_u16(clamp_u16_from_float(MAV_CV_WEIGHT_ORANGE * (float)orange.middle +
                                                        MAV_CV_WEIGHT_GREEN * (float)green.middle +
                                                        MAV_CV_WEIGHT_BLUE * (float)blue.middle));
  out.right = clamp_i16_range_u16(clamp_u16_from_float(MAV_CV_WEIGHT_ORANGE * (float)orange.right +
                                                       MAV_CV_WEIGHT_GREEN * (float)green.right +
                                                       MAV_CV_WEIGHT_BLUE * (float)blue.right));
  return out;
}