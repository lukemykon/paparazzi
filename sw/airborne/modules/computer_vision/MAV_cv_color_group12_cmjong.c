/**
 * @file MAV_cv_color_group12_cmjong.c
 * @brief YUV422 color detection utilities: pixel counting and left/center/right column scoring.
 *
 * COORDINATE SYSTEM NOTE — the Bebop front camera feed is rotated 90° clockwise before
 * it reaches these functions.  As a result the image axes are swapped relative to the
 * physical scene:
 *   image x  →  physical vertical  (top-to-bottom in the frame)
 *   image y  →  physical left-right (left wing to right wing of the drone)
 * All "column" boundaries in color_detection_columns() therefore partition the image
 * along the y-axis (rows in buffer terms) to produce left / center / right scores.
 */

#include "modules/computer_vision/MAV_cv_color_group12_cmjong.h"
#define SCAN_NUM_LINES   5   ///< number of vertical scan strips (in image-x)
#define SCAN_THICKNESS   1   ///< width of each strip in pixels
#define SCAN_SPACING     15  ///< gap between adjacent strip edges in pixels

/**
 * draw_horizontal_line — paint a single-pixel-wide vertical line at column x_col
 * into a YUV422 buffer.
 *
 *
 * @param buffer  Pointer to the YUV422 image buffer (in-place modification).
 * @param img_w   Image width in pixels.
 * @param img_h   Image height in pixels.
 * @param x_col   Column index (image-x) to draw; rounded down to an even pixel.
 * @param y_val   Luma (Y) to write for the line pixels.
 * @param u_val   Cb (U) chroma to write.
 * @param v_val   Cr (V) chroma to write.
 */
static void draw_horizontal_line(uint8_t *buffer, uint16_t img_w, uint16_t img_h,
                                uint16_t x_col, uint8_t y_val, uint8_t u_val, uint8_t v_val)
{
  if (!buffer || img_w < 2 || x_col >= img_w) {
    return;
  }

  uint16_t x_even = x_col & ~1u;
  if (x_even + 1 >= img_w) {
    x_even = img_w - 2;
  }

  for (uint16_t y = 0; y < img_h; y++) {
    uint32_t base = y * 2u * img_w + 2u * x_even;
    buffer[base]     = u_val;
    buffer[base + 1] = y_val;
    buffer[base + 2] = v_val;
    buffer[base + 3] = y_val;
  }
}

/**
 * color_detection — count pixels in the image that fall within a YUV color range.
 *
 * Samples only the sparse scan strips defined by SCAN_NUM_LINES / SCAN_THICKNESS /
 * SCAN_SPACING rather than the full frame, to reduce CPU load.
 *
 * @param img      Source YUV422 image.  Modified in-place only when draw == true.
 * @param lum_min/lum_max  Inclusive luma (Y) range.
 * @param cb_min/cb_max    Inclusive Cb (U) range.
 * @param cr_min/cr_max    Inclusive Cr (V) range.
 * @param draw     If true, overlay white scan-band boundary lines on img->buf.
 * @return Raw pixel count of matching pixels across all scan strips (capped at 65535).
 */
uint16_t color_detection(struct image_t *img,
                         uint8_t lum_min, uint8_t lum_max,
                         uint8_t cb_min,  uint8_t cb_max,
                         uint8_t cr_min,  uint8_t cr_max,
                         bool draw)
{
  uint32_t total   = 0;
  uint8_t *buffer  = img->buf;
  uint16_t w       = img->w;
  uint16_t h       = img->h;
  // YUV422 stores pixels in even/odd pairs; truncate to even width to avoid partial macro-pixels.
  uint16_t scan_w  = w & ~1u;

  if (!buffer || scan_w < 2 || h == 0) {
    return 0;
  }

  uint16_t block    = SCAN_THICKNESS + SCAN_SPACING;
  uint16_t total_w  = block * SCAN_NUM_LINES - SCAN_SPACING;
  uint16_t x_offset = (scan_w > total_w) ? (scan_w - total_w) / 2 : 0;

  uint16_t band_start[SCAN_NUM_LINES];
  uint16_t band_end[SCAN_NUM_LINES];

  for (uint8_t i = 0; i < SCAN_NUM_LINES; i++) {
    band_start[i] = x_offset + i * block;
    band_end[i] = band_start[i] + SCAN_THICKNESS;

    if (band_start[i] >= scan_w) {
      band_start[i] = scan_w;
      band_end[i] = scan_w;
      continue;
    }
    if (band_end[i] > scan_w) {
      band_end[i] = scan_w;
    }
  }

  // VLA: stack-allocated lookup for each column; size = scan_w (typically 240 or 320 pixels).
  // scan_w is always even and bounded by the camera resolution so stack usage is safe.
  bool in_band[scan_w];
  for (uint16_t x = 0; x < scan_w; x++) {
    in_band[x] = false;
    for (uint8_t i = 0; i < SCAN_NUM_LINES; i++) {
      if (x >= band_start[i] && x < band_end[i]) {
        in_band[x] = true;
        break;
      }
    }
  }

  for (uint16_t y = 0; y < h; y++) {
    for (uint16_t x = 0; x < scan_w; x++) {
      if (!in_band[x]) {
        continue;
      }

      uint32_t pair_base = y * 2u * w + 2u * (x & ~1u);
      uint8_t *up = &buffer[pair_base];
      uint8_t *vp = &buffer[pair_base + 2u];
      uint8_t *yp = &buffer[pair_base + ((x & 1u) ? 3u : 1u)];

      if ((*yp >= lum_min) && (*yp <= lum_max) &&
          (*up >= cb_min)  && (*up <= cb_max)  &&
          (*vp >= cr_min)  && (*vp <= cr_max)) {
        total++;
      }
    }
  }

  if (draw) {
    for (uint8_t i = 0; i < SCAN_NUM_LINES; i++) {
      if (band_start[i] >= scan_w || band_end[i] <= band_start[i]) {
        continue;
      }
      draw_horizontal_line(buffer, w, h, band_start[i], 235, 128, 128);
      draw_horizontal_line(buffer, w, h, band_end[i] - 1u, 235, 128, 128);
    }
  }

  return (uint16_t)(total > 65535 ? 65535 : total);
}

/*
 * Column boundary fractions along the image y-axis (= physical left/right, see file header).
 *
 * The three regions intentionally overlap by 5% on each shared edge so that an obstacle
 * near a boundary scores in both adjacent regions — this prevents blind spots and gives
 * the controller a smoother directional signal when an obstacle straddles a boundary.
 *
 *   Physical:   LEFT       CENTER          RIGHT
 *   y fraction: 0 ─── 30%  25% ─── 75%  70% ─── 100%
 *               └──── 5% overlap ──┘  └── 5% overlap ──┘
 */
#define COL_LEFT_FRAC    0.30f   ///< upper boundary of the LEFT zone (fraction of image h)
#define COL_CENTER_START 0.25f   ///< start of CENTER zone (overlaps LEFT by 5%)
#define COL_CENTER_END   0.75f   ///< end of CENTER zone (overlaps RIGHT by 5%)
#define COL_RIGHT_FRAC   0.70f   ///< lower boundary of the RIGHT zone

/**
 * draw_row_line — paint a single-pixel-high horizontal line at row y_row.
 *
 * Despite the "row" name, because the camera is rotated 90° this line appears as a
 * vertical column boundary in the physical scene.  Used to overlay the left/center/right
 * zone boundaries on the streamed video for debugging.
 */
static void draw_row_line(uint8_t *buffer, uint16_t img_w, uint16_t img_h,
                          uint16_t y_row, uint8_t y_val, uint8_t u_val, uint8_t v_val)
{
  if (!buffer || img_w < 2 || y_row >= img_h) return;
  uint32_t row_base = (uint32_t)y_row * 2u * img_w;
  for (uint16_t x = 0; x < (img_w & ~1u); x += 2) {
    uint32_t base = row_base + 2u * x;
    buffer[base]     = u_val;
    buffer[base + 1] = y_val;
    buffer[base + 2] = v_val;
    buffer[base + 3] = y_val;
  }
}

/**
 * color_detection_columns — count matching pixels per physical zone (LEFT/CENTER/RIGHT).
 *
 * Partitions the frame into three overlapping zones along the image y-axis (physical
 * left/right direction — see file-level coordinate note).  Uses the same sparse scan-band
 * sampling as color_detection() to limit CPU usage.
 *
 * @param img      Source YUV422 image.  Modified in-place only when draw == true.
 * @param lum_min/lum_max  Inclusive luma (Y) range.
 * @param cb_min/cb_max    Inclusive Cb (U) range.
 * @param cr_min/cr_max    Inclusive Cr (V) range.
 * @param draw     If true, overlay scan-band and zone-boundary lines on img->buf.
 * @return struct column_counts with raw pixel counts for left, center, and right zones.
 *         Zones overlap, so a pixel near a boundary may contribute to two counts.
 */
struct column_counts color_detection_columns(struct image_t *img,
                                             uint8_t lum_min, uint8_t lum_max,
                                             uint8_t cb_min,  uint8_t cb_max,
                                             uint8_t cr_min,  uint8_t cr_max,
                                             bool draw)
{
  struct column_counts counts = {0, 0, 0};
  uint8_t *buffer = img->buf;
  uint16_t w      = img->w;
  uint16_t h      = img->h;
  uint16_t scan_w = w & ~1u;

  if (!buffer || scan_w < 2 || h == 0) return counts;

  // Column boundaries along h (y-axis = physical left-right)
  uint16_t left_end     = (uint16_t)(h * COL_LEFT_FRAC);
  uint16_t center_start = (uint16_t)(h * COL_CENTER_START);
  uint16_t center_end   = (uint16_t)(h * COL_CENTER_END);
  uint16_t right_start  = (uint16_t)(h * COL_RIGHT_FRAC);

  // Scan bands along x (same as color_detection)
  uint16_t block    = SCAN_THICKNESS + SCAN_SPACING;
  uint16_t total_w  = block * SCAN_NUM_LINES - SCAN_SPACING;
  uint16_t x_offset = (scan_w > total_w) ? (scan_w - total_w) / 2 : 0;

  uint16_t bs[SCAN_NUM_LINES], be[SCAN_NUM_LINES];
  for (uint8_t i = 0; i < SCAN_NUM_LINES; i++) {
    bs[i] = x_offset + i * block;
    be[i] = bs[i] + SCAN_THICKNESS;
    if (bs[i] >= scan_w) { bs[i] = scan_w; be[i] = scan_w; continue; }
    if (be[i] > scan_w) be[i] = scan_w;
  }

  // VLA: same stack-based column lookup as in color_detection(); safe for typical resolutions.
  bool x_in_band[scan_w];
  for (uint16_t x = 0; x < scan_w; x++) {
    x_in_band[x] = false;
    for (uint8_t i = 0; i < SCAN_NUM_LINES; i++) {
      if (x >= bs[i] && x < be[i]) { x_in_band[x] = true; break; }
    }
  }

  for (uint16_t y = 0; y < h; y++) {
    for (uint16_t x = 0; x < scan_w; x++) {
      if (!x_in_band[x]) continue;

      uint32_t pair_base = y * 2u * w + 2u * (x & ~1u);
      uint8_t *up = &buffer[pair_base];
      uint8_t *vp = &buffer[pair_base + 2u];
      uint8_t *yp = &buffer[pair_base + ((x & 1u) ? 3u : 1u)];

      if ((*yp >= lum_min) && (*yp <= lum_max) &&
          (*up >= cb_min)  && (*up <= cb_max)  &&
          (*vp >= cr_min)  && (*vp <= cr_max)) {
        if (y < left_end)                         counts.left++;
        if (y >= center_start && y < center_end)  counts.center++;
        if (y >= right_start)                      counts.right++;
      }
    }
  }

  if (draw) {

    for (uint8_t i = 0; i < SCAN_NUM_LINES; i++) {
      if (bs[i] >= scan_w || be[i] <= bs[i]) continue;
      draw_horizontal_line(buffer, w, h, bs[i], 235, 128, 128);      // band start
      draw_horizontal_line(buffer, w, h, be[i] - 1u, 235, 128, 128); // band end
    }
    // Draw column boundary lines (horizontal lines in buffer = vertical in physical scene)
    draw_row_line(buffer, w, h, left_end,     0, 200, 128);   // left|center boundary (blue-ish)
    draw_row_line(buffer, w, h, center_start, 0, 200, 128);   // center overlap start
    draw_row_line(buffer, w, h, center_end,   0, 128, 200);   // center|right boundary (red-ish)
    draw_row_line(buffer, w, h, right_start,  0, 128, 200);   // right overlap start
  }

  return counts;
}
