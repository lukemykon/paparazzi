#ifndef SQUARE_PASSER_GUIDED_H
#define SQUARE_PASSER_GUIDED_H

#include <stdbool.h>

extern float spg_forward_speed;
extern float spg_max_lateral_speed;
extern float spg_lateral_gain;
extern float spg_search_heading_rate;
extern float spg_search_forward_speed;
extern float spg_center_forward_speed;
extern float spg_center_confirm_time;
extern float spg_vertical_gain;
extern float spg_max_vertical_step;
extern float spg_search_vertical_rate;
extern float spg_search_min_altitude;
extern float spg_search_max_altitude;
extern float spg_detection_timeout;
extern float spg_gate_pass_distance;
extern float spg_gate_center_tolerance;
extern float spg_gate_vertical_tolerance;
extern float spg_pass_through_time;
extern float spg_min_guided_altitude;
extern float spg_camera_delay_comp_s;
extern float spg_max_prediction_horizon_s;
extern float spg_max_relative_velocity;

extern void square_passer_guided_init(void);
extern void square_passer_guided_periodic(void);
extern void square_passer_guided_set_enabled(bool enabled);
extern bool square_passer_guided_is_enabled(void);

#endif
