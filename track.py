import cv2
import numpy as np
import struct
import time
import matplotlib.pyplot as plt
import os

##############################################################################
#                          CINE File Reading Functions                       #
##############################################################################

def read_L(f):
    return int(struct.unpack('<l', f.read(4))[0])

def read_Q(f):
    return struct.unpack('Q', f.read(8))[0]

def read_Q_array(f, n):
    a = np.zeros(n, dtype='Q')
    for i in range(n):
        a[i] = read_Q(f)
    return a

def read_B_2Darray(f, ypix, xpix):
    n = xpix * ypix
    a = np.array(struct.unpack(f'{n}B', f.read(n * 1)), dtype='B')
    return a.reshape(ypix, xpix)

def read_H_2Darray(f, ypix, xpix):
    n = xpix * ypix
    a = np.array(struct.unpack(f'{n}H', f.read(n * 2)), dtype='H')
    return a.reshape(ypix, xpix)

def read_cine(ifn):
    with open(ifn, 'rb') as cf:
        t_read = time.time()
        print("Reading .cine file...")

        cf.read(16)
        baseline_image = read_L(cf)
        image_count = read_L(cf)

        pointers = np.zeros(3, dtype='L')
        pointers[0] = read_L(cf)
        pointers[1] = read_L(cf)
        pointers[2] = read_L(cf)

        cf.seek(58)
        nbit = read_L(cf)

        cf.seek(int(pointers[0]) + 4)
        xpix = read_L(cf)
        ypix = read_L(cf)

        cf.seek(int(pointers[1]) + 768)
        pps = read_L(cf)
        exposure = read_L(cf)

        cf.seek(int(pointers[2]))
        pimage = read_Q_array(cf, image_count)

        dtype = 'B' if nbit == 8 else 'H'
        frame_arr = np.zeros((image_count, ypix, xpix), dtype=dtype)

        for i in range(image_count):
            p = struct.unpack('<l', struct.pack('<L', pimage[i] & 0xffffffffffffffff))[0]
            cf.seek(p)
            ofs = read_L(cf)
            cf.seek(p + ofs)
            frame_arr[i] = read_B_2Darray(cf, ypix, xpix) if nbit == 8 else read_H_2Darray(cf, ypix, xpix)

        time_arr = np.linspace(
            baseline_image / pps, 
            (baseline_image + image_count) / pps, 
            image_count, 
            endpoint=False
        )

        print("Done reading .cine file (%.1f s)" % (time.time() - t_read))
        return time_arr, frame_arr

##############################################################################
#                          Video Processing Functions                        #
##############################################################################

def convert_cine_to_avi(frame_arr, avi_path, scale_factor=8):
    """Convert CINE frame array to AVI video"""
    orig_height, orig_width = frame_arr.shape[1], frame_arr.shape[2]
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(avi_path, fourcc, 30, 
                        (orig_width*scale_factor, orig_height*scale_factor), False)

    print(f"Converting to {avi_path}...")
    for frame in frame_arr:
        norm_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        resized = cv2.resize(norm_frame.astype(np.uint8), 
                           (orig_width*scale_factor, orig_height*scale_factor))
        flipped = cv2.flip(resized, 0)  # Flip vertically
        out.write(flipped)
    
    out.release()
    print(f"Conversion complete. Saved to {avi_path}")

def detect_chamber(frame, calibration):
    """
    Detects the bright chamber circle using optimized thresholding and validation.
    """
    # Convert physical radius constraints (6-8cm in pixels)
    min_radius_px = int(6 / calibration)
    max_radius_px = int(8 / calibration)

    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Optimized preprocessing for bright circles
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological closing to enhance circular shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Detect circles with optimized parameters
    circles = cv2.HoughCircles(
        closed,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=frame.shape[1]//2,  # Assume only one main chamber
        param1=150,  # Lower Canny threshold
        param2=25,   # Accumulator threshold (lower for better detection)
        minRadius=min_radius_px,
        maxRadius=max_radius_px
    )

    # Validate and select best candidate
    best_circle = None
    if circles is not None:
        circles = np.int32(np.around(circles))[0]
        
        # Score circles by brightness and circularity
        for circle in circles:
            x, y, r = circle
            if x-r < 0 or y-r < 0 or x+r > frame.shape[1] or y+r > frame.shape[0]:
                continue  # Skip edge-touching circles
            
            # Create mask for brightness verification
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r, 255, -1)
            mean_brightness = cv2.mean(gray, mask=mask)[0]
            
            # Circularity check
            contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            perimeter = cv2.arcLength(contour[0], True)
            circularity = 4 * np.pi * (cv2.contourArea(contour[0])) / (perimeter ** 2)
            
            if circularity > 0.85 and mean_brightness > 200:
                if best_circle is None or r > best_circle[2]:
                    best_circle = (x, y, r)

    # Fallback to contour detection if Hough fails
    if best_circle is None:
        print("Hough failed, using contour fallback")
        contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(largest_contour)
            best_circle = (int(x), int(y), int(r))

    # Final validation
    if best_circle:
        x, y, r = best_circle
        origin = (x, y)
        radius = r
    else:
        print("Warning: No valid circle found, using frame center")
        origin = (frame.shape[1]//2, frame.shape[0]//2)
        radius = int((min_radius_px + max_radius_px)/2)

    # Visualization (save intermediate steps)
    debug_img = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    cv2.circle(debug_img, origin, radius, (0,255,0), 2)
    cv2.imwrite("chamber_debug.png", debug_img)

    print(f"Chamber detected at {origin} with radius {radius}px")
    return origin, radius
def track_object(avi_path, output_prefix, time_arr):
    """Track tungsten ball through entire video sequence"""
    # Constants
    CALIBRATION = 0.038  # cm/pixel
    MIN_DIAMETER_MM = 1   # Tungsten ball diameter
    MAX_DIAMETER_MM = 3
    PIXELS_PER_METER = 100 / CALIBRATION  # Assuming CALIBRATION is in cm/pixel

    # Initialize video capture
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {avi_path}")

    # Detect chamber in first frame
    ret, initial_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame for chamber detection")

    (cx, cy), chamber_radius = detect_chamber(initial_frame, PIXELS_PER_METER)
    
    # Save chamber visualization
    chamber_vis = initial_frame.copy()
    cv2.circle(chamber_vis, (cx, cy), chamber_radius, (0, 255, 0), 2)
    cv2.imwrite(f"{output_prefix}_chamber_detection.png", chamber_vis)

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
        plt.plot(x, y, '.-', markersize=3)
        plt.gca().invert_yaxis()
        plt.title('Trajectory in Chamber')
        plt.xlabel('X Position (cm)')
        plt.ylabel('Y Position (cm)')
        plt.grid(True)
        
        # Vertical motion plot
        plt.subplot(132)
        plt.plot(times, y, '.-', markersize=3)
        plt.gca().invert_yaxis()
        plt.title('Vertical Motion')
        plt.xlabel('Time (s)')
        plt.ylabel('Y Position (cm)')
        plt.grid(True)
        
        # Horizontal motion plot
        plt.subplot(133)
        plt.plot(times, x, '.-', markersize=3)
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
