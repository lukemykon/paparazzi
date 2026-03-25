#ifndef MAV_BLUE_DETECTOR_H
#define MAV_BLUE_DETECTOR_H

#include "std.h"
#include "modules/computer_vision/lib/vision/image.h"

#ifdef __cplusplus
extern "C" {
#endif

extern float bd_load_left;
extern float bd_load_straight;
extern float bd_load_right;
extern float bd_forward_speed;
extern float bd_turn_speed;
extern int8_t bd_last_direction;

extern float bd_weight_left;
extern float bd_weight_straight;
extern float bd_weight_right;

void blue_detector_detect_losses(struct image_t *img, bool draw_mask,
                                 uint32_t *left, uint32_t *straight, uint32_t *right);

void blue_detector_init(void);
void blue_detector_periodic(void);

#ifdef __cplusplus
}
#endif

#endif
