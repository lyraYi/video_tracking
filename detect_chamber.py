import cv2
import numpy as np
import struct
import time
import matplotlib.pyplot as plt
import os
import re

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
def track_object(avi_path, output_prefix, time_arr, CALIBRATION, cx, cy, chamber_radius):
