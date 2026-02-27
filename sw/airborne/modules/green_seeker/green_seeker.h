/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/green_seeker/green_seeker.h"
 * @author Roland Meertens (Adapted for Green Seeking)
 * Example on how to use the colours detected to track a green pole in the cyberzoo
 */

#ifndef GREEN_SEEKER_H
#define GREEN_SEEKER_H

#include "std.h"

/* * Threshold fraction of green pixels required to trigger tracking.
 * Exposed here so it can be dynamically tuned from the Ground Control Station (GCS)
 * via the settings XML.
 */
extern float gs_color_count_frac;

/*
 * Initialisation function. Binds the ABI message to receive color filter outputs.
 */
extern void green_seeker_init(void);

/*
 * Periodic function. Evaluates the state machine to track green or search for it.
 */
extern void green_seeker_periodic(void);

#endif /* GREEN_SEEKER_H */