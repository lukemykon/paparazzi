#include "MAV_cv_color_gate_group12_rghendriks.h"
#include <stdlib.h>

// Mask dimensions matching the raw, physical camera memory (Portrait orientation)
#define GATE_W 120
#define GATE_H 160
#define MAX_BLOBS 10
#define MAX_STACK 4000

typedef struct {
    uint16_t x, y, w, h;
    uint16_t cx, cy;
    uint16_t area;
} Blob;

// Arrays for the flood fill stack (static to avoid stack overflows on embedded targets)
static uint8_t stack_r[MAX_STACK];
static uint8_t stack_c[MAX_STACK];

GateResult detect_gate(struct image_t *img, 
                       uint8_t lum_min, uint8_t lum_max,
                       uint8_t cb_min,  uint8_t cb_max,
                       uint8_t cr_min,  uint8_t cr_max) 
{
    GateResult result = {false, 0, 0};
    uint8_t *buffer = img->buf;
    
    // 120x160 binary mask (0 = background, 1 = blue, 2 = visited)
    static uint8_t mask[GATE_H][GATE_W]; 
    
    // 1. DOWNSCALE & THRESHOLD (Linear Memory Access - Cache Friendly)
    for (uint16_t r = 0; r < GATE_H; r++) {
        for (uint16_t c = 0; c < GATE_W; c++) {
            uint16_t orig_x = c * img->w / GATE_W;
            uint16_t orig_y = r * img->h / GATE_H;
            
            // Safety bounds check
            if (orig_x >= img->w) orig_x = img->w - 1;
            if (orig_y >= img->h) orig_y = img->h - 1;

            // Extract YUV values based on YUV422 memory layout
            uint8_t *yp, *up, *vp;
            if (orig_x % 2 == 0) {
                up = &buffer[orig_y * 2 * img->w + 2 * orig_x];
                yp = &buffer[orig_y * 2 * img->w + 2 * orig_x + 1];
                vp = &buffer[orig_y * 2 * img->w + 2 * orig_x + 2];
            } else {
                up = &buffer[orig_y * 2 * img->w + 2 * orig_x - 2];
                vp = &buffer[orig_y * 2 * img->w + 2 * orig_x];
                yp = &buffer[orig_y * 2 * img->w + 2 * orig_x + 1];
            }

            // Apply Blue thresholds
            if ((*yp >= lum_min) && (*yp <= lum_max) &&
                (*up >= cb_min)  && (*up <= cb_max)  &&
                (*vp >= cr_min)  && (*vp <= cr_max)) {
                mask[r][c] = 1;
            } else {
                mask[r][c] = 0;
            }
        }
    }

    // 2. BLOB DETECTION (Non-recursive Flood Fill)
    Blob blobs[MAX_BLOBS];
    int num_blobs = 0;

    for (uint16_t r = 0; r < GATE_H; r++) {
        for (uint16_t c = 0; c < GATE_W; c++) {
            if (mask[r][c] == 1) {
                uint16_t min_x = c, max_x = c, min_y = r, max_y = r;
                uint16_t area = 0;
                
                int stack_ptr = 0;
                stack_r[stack_ptr] = r;
                stack_c[stack_ptr] = c;
                stack_ptr++;
                mask[r][c] = 2; // mark visited

                while (stack_ptr > 0 && stack_ptr < MAX_STACK) {
                    stack_ptr--;
                    uint8_t curr_r = stack_r[stack_ptr];
                    uint8_t curr_c = stack_c[stack_ptr];
                    area++;
                    
                    if (curr_c < min_x) min_x = curr_c;
                    if (curr_c > max_x) max_x = curr_c;
                    if (curr_r < min_y) min_y = curr_r;
                    if (curr_r > max_y) max_y = curr_r;

                    // 4-way neighbors
                    int dr[] = {-1, 1, 0, 0};
                    int dc[] = {0, 0, -1, 1};
                    for (int i = 0; i < 4; i++) {
                        int nr = curr_r + dr[i];
                        int nc = curr_c + dc[i];
                        if (nr >= 0 && nr < GATE_H && nc >= 0 && nc < GATE_W) {
                            if (mask[nr][nc] == 1) {
                                mask[nr][nc] = 2;
                                if (stack_ptr < MAX_STACK) {
                                    stack_r[stack_ptr] = nr;
                                    stack_c[stack_ptr] = nc;
                                    stack_ptr++;
                                }
                            }
                        }
                    }
                }

                uint16_t w = max_x - min_x + 1;
                uint16_t h = max_y - min_y + 1;
                
                // 3. FAST MATH FILTERING (Rotated Logic)
                // Because the camera is rotated 90 deg CW, vertical pillars look like wide horizontal bars.
                if (area >= 30 && (w > h * 2) && (w < h * 15) && h >= 1) {
                    if (num_blobs < MAX_BLOBS) {
                        blobs[num_blobs].x = min_x;
                        blobs[num_blobs].y = min_y;
                        blobs[num_blobs].w = w;
                        blobs[num_blobs].h = h;
                        blobs[num_blobs].cx = min_x + w / 2;
                        blobs[num_blobs].cy = min_y + h / 2;
                        blobs[num_blobs].area = area;
                        num_blobs++;
                    }
                }
            }
        }
    }

    // 4. SORT BLOBS BY AREA (Largest First)
    for (int i = 0; i < num_blobs - 1; i++) {
        for (int j = 0; j < num_blobs - i - 1; j++) {
            if (blobs[j].area < blobs[j + 1].area) {
                Blob temp = blobs[j];
                blobs[j] = blobs[j + 1];
                blobs[j + 1] = temp;
            }
        }
    }

    // 5. PAIRING AND CONFIDENCE CALCULATION (Rotated Logic)
    int best_confidence = 0;
    int img_cy = GATE_H / 2; // Center of image along Y axis

    for (int j = 0; j < num_blobs; j++) {
        for (int k = j + 1; k < num_blobs; k++) {
            Blob *b1 = &blobs[j];
            Blob *b2 = &blobs[k];
            
            int min_area = b1->area < b2->area ? b1->area : b2->area;
            int max_area = b1->area > b2->area ? b1->area : b2->area;
            
            // In rotated space, 'width' of the bar represents physical height of the pillar
            int min_w = b1->w < b2->w ? b1->w : b2->w;
            int max_w = b1->w > b2->w ? b1->w : b2->w;
            
            int dx = abs(b1->cx - b2->cx); // Separation along image X (physical Y / Altitude)
            int dy = abs(b1->cy - b2->cy); // Separation along image Y (physical X / Horizontal)
            
            // Filter pairs based on rotated physical properties
            if ((min_area * 10 > max_area * 1) &&
                (min_w * 4 > max_w * 1) &&
                (dx * 2 < max_w * 3) &&  // Pillars must be roughly aligned vertically in physical space
                (dy > dx) &&             // Pillars must be separated horizontally in physical space
                (dy < min_w * 5)) {
                
                int height_score = (min_w * 100) / max_w;
                int area_score = (min_area * 100) / max_area;
                int align_penalty = (dx * 100) / max_w;
                int align_score = align_penalty < 100 ? 100 - align_penalty : 0;
                
                int size_score = (max_w * 100) / 60;
                if (size_score > 100) size_score = 100;
                
                int pair_confidence = (height_score + area_score + align_score + size_score) / 4;
                
                if (pair_confidence > best_confidence) {
                    best_confidence = pair_confidence;
                    result.found = true;
                    
                    int gate_cy = (b1->cy + b2->cy) / 2;
                    
                    // Steer X logic mapping:
                    // Camera is rotated 90 CW. The physical top points to image left (x=0). 
                    // The physical right points to image top (y=0).
                    // If gate_cy is less than img_cy (closer to y=0), the gate is physically to the RIGHT.
                    // Therefore: img_cy - gate_cy gives a POSITIVE value when the gate is to the right.
                    result.steering_x = img_cy - gate_cy; 
                    result.confidence = pair_confidence;
                }
            }
        }
    }

    return result;
}