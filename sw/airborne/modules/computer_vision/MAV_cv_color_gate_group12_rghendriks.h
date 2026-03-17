#ifndef MAV_CV_COLOR_GATE_GROUP12_RGHENDRIKS_H_
#define MAV_CV_COLOR_GATE_GROUP12_RGHENDRIKS_H_

#include <stdint.h>
#include <stdbool.h>
#include "lib/vision/image.h"

typedef struct {
    bool found;
    int16_t steering_x; 
    uint8_t confidence; 
} GateResult;
/**
 * detect_gate
 *
 * Scans a 90-degree CW rotated image for blue pillars (which appear as horizontal bars).
 * Employs linear memory reading to prevent CPU cache misses.
 */
GateResult detect_gate(struct image_t *img, 
                       uint8_t lum_min, uint8_t lum_max,
                       uint8_t cb_min,  uint8_t cb_max,
                       uint8_t cr_min,  uint8_t cr_max);

#endif