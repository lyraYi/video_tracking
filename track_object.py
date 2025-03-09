import cv2
import numpy as np
import struct
import time
import matplotlib.pyplot as plt
import os
def track_object(avi_path, output_prefix, time_arr, CALIBRATION, cx, cy, chamber_radius):
    """Track tungsten ball through entire video sequence"""
    # Constants
    MIN_DIAMETER_MM = 1   # Tungsten ball diameter
    MAX_DIAMETER_MM = 3
    PIXELS_PER_METER = 100 / CALIBRATION  # Assuming CALIBRATION is in cm/pixel

    # Initialize video capture
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {avi_path}")
    # Prepare tracking data structures
    positions = []
    frame_numbers = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process all frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Detect ball position
        mask = np.zeros_like(frame[:,:,0])
        cv2.circle(mask, (cx, cy), chamber_radius, 255, -1)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        inverted = 255 - blurred

        # Correct radius calculation (using radius instead of diameter)
        min_radius = int((MIN_DIAMETER_MM/10)/CALIBRATION)  # 0.1cm
        max_radius = int((MAX_DIAMETER_MM/10)/CALIBRATION)  # 0.3cm
        
        circles = cv2.HoughCircles(
            inverted,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=chamber_radius//4,
            param1=50,
            param2=12,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        if circles is not None:
            circles = np.int32(np.around(circles[0]))
            valid = []
            for c in circles:
                # Validate position within chamber
                if np.hypot(c[0] - cx, c[1] - cy) < chamber_radius * 0.9:
                    valid.append(c)
            
            if valid:
                # Select brightest candidate
                brightest = max(valid, key=lambda c: gray[c[1], c[0]])
                px, py, radius = brightest
                
                # Convert to chamber-relative coordinates
                rel_x = (px - cx) * CALIBRATION
                rel_y = (cy - py) * CALIBRATION  # Inverted Y-axis
                positions.append((rel_x, rel_y))
                frame_numbers.append(frame_idx)

        # Visualization every 50 frames
        if frame_idx % 30 == 0:
            vis_frame = frame.copy()
            if positions:
                cv2.circle(vis_frame, (px, py), radius * 2, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"({rel_x:.2f}, {rel_y:.2f} cm)",
                           (px + 20, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(f"{output_prefix}_frame_{frame_idx:04d}.png", vis_frame)

    # Analysis plots
    if len(positions) > 1:
        x = [p[0] for p in positions]
        y = [p[1] for p in positions]
        times = time_arr[frame_numbers]
        
        plt.figure(figsize=(15, 5))
        
        # Trajectory plot
        plt.subplot(131)
        plt.scatter(x, y)
        plt.gca().invert_yaxis()
        plt.title('Trajectory in Chamber')
        plt.xlabel('X Position (cm)')
        plt.ylabel('Y Position (cm)')
        plt.grid(True)
        
        # Vertical motion plot
        plt.subplot(132)
        plt.scatter(times, y)
        plt.gca().invert_yaxis()
        plt.title('Vertical Motion')
        plt.xlabel('Time (s)')
        plt.ylabel('Y Position (cm)')
        plt.grid(True)
        
        # Horizontal motion plot
        plt.subplot(133)
        plt.scatter(times, x)
        plt.title('Horizontal Motion')
        plt.xlabel('Time (s)')
        plt.ylabel('X Position (cm)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_full_analysis.png")
        plt.close()

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {len(positions)} positions")
    return np.array(positions), np.array(times)
