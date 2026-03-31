import cv2
import os
import glob
import numpy as np

folder_path = "training_img"
image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

if not image_paths:
    print(f"Error: No images found in '{folder_path}'. Please check the folder path.")
    exit()

print(f"Found {len(image_paths)} images. Press ANY KEY to advance, 'q' or 'ESC' to quit.")

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue
        
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # 1. DOWNSCALE
    W_TINY, H_TINY = 160, 120
    img_tiny = cv2.resize(img, (W_TINY, H_TINY), interpolation=cv2.INTER_LINEAR)
    img_cx = W_TINY // 2
    img_cy = H_TINY // 2
    
    # 2. COLOR ISOLATION
    hsv = cv2.cvtColor(img_tiny, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 80, 50]) 
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 3. BLOB DETECTION (In C, this will be your Connected Components loop)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    
    # 4. CANDIDATE FILTERING
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h 
        
        # Must be large enough, and shaped like a vertical pillar
        if area >= 30 and (h > w * 2) and (h < w * 15) and w >= 1: 
            candidates.append({'x': x, 'y': y, 'w': w, 'h': h, 'cx': x + w//2, 'cy': y + h//2, 'area': area})

    # ---------------------------------------------------------
    # STATE MACHINE & CONFIDENCE
    # ---------------------------------------------------------
    gate_found = False
    steering_x = 0
    best_confidence = 0
    
    best_b1 = None
    best_b2 = None
    gate_cx = img_cx
    gate_cy = img_cy # Tracked solely for the visualizer drawing
    
    # 5. PAIR MATCHING LOOP
    for j in range(len(candidates)):
        for k in range(j + 1, len(candidates)):
            b1 = candidates[j]
            b2 = candidates[k]
            
            min_area, max_area = min(b1['area'], b2['area']), max(b1['area'], b2['area'])
            min_h, max_h = min(b1['h'], b2['h']), max(b1['h'], b2['h'])
            dx = abs(b1['cx'] - b2['cx'])
            dy = abs(b1['cy'] - b2['cy'])
            
            # Geometric constraints
            if (min_area * 10 > max_area * 1) and \
               (min_h * 4 > max_h * 1) and \
               (dy * 2 < max_h * 3) and \
               (dx > dy) and \
               (dx < min_h * 5):
               
                # Calculate integer confidence
                height_score = (min_h * 100) // max_h
                area_score = (min_area * 100) // max_area
                align_score = max(0, 100 - ((dy * 100) // max_h))
                size_score = min(100, (max_h * 100) // 60)
                
                pair_confidence = (height_score + area_score + align_score + size_score) // 4
                
                # Update if this is the best pair seen so far
                if pair_confidence > best_confidence:
                    best_confidence = pair_confidence
                    gate_found = True
                    best_b1, best_b2 = b1, b2
                    
                    gate_cx = (b1['cx'] + b2['cx']) // 2
                    gate_cy = (b1['cy'] + b2['cy']) // 2 
                    steering_x = gate_cx - img_cx

    # ---------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------
    color = (0, 255, 0) if gate_found else (0, 0, 255)
    state_text = f"TRACK (Conf:{best_confidence}%) X:{steering_x}" if gate_found else "SEARCHING"

    if gate_found:
        # Draw the gate and steering line
        cv2.rectangle(img_tiny, (best_b1['x'], best_b1['y']), (best_b1['x']+best_b1['w'], best_b1['y']+best_b1['h']), color, 2)
        cv2.rectangle(img_tiny, (best_b2['x'], best_b2['y']), (best_b2['x']+best_b2['w'], best_b2['y']+best_b2['h']), color, 2)
        cv2.circle(img_tiny, (gate_cx, gate_cy), 4, color, -1)
        cv2.line(img_tiny, (img_cx, img_cy), (gate_cx, gate_cy), color, 1)
    else:
        # Draw ignored yellow candidates and a red X
        for c in candidates:
            cv2.rectangle(img_tiny, (c['x'], c['y']), (c['x']+c['w'], c['y']+c['h']), (0, 255, 255), 1) 
        cv2.line(img_tiny, (img_cx-5, img_cy-5), (img_cx+5, img_cy+5), color, 2)
        cv2.line(img_tiny, (img_cx-5, img_cy+5), (img_cx+5, img_cy-5), color, 2)

    # Scale up and display
    display_img = cv2.resize(img_tiny, (640, 480), interpolation=cv2.INTER_NEAREST)
    cv2.putText(display_img, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Gate Vision", display_img)
    
    if cv2.waitKey(0) & 0xFF in [27, ord('q')]: 
        break

cv2.destroyAllWindows()


