import cv2
import numpy as np
import struct
import time
import matplotlib.pyplot as plt
import os
import re

if __name__ == '__main__':
    # Define input folder
    input_folder = "He3kA_B380G800G_pl0t20_uw15t45"
    output_root = "output"
    os.makedirs(output_root, exist_ok=True)

    # Get all .cine files in the input folder
    cine_files = [f for f in os.listdir(input_folder) if f.endswith(".cine")]

    if not cine_files:
        print("No .cine files found in the folder.")
        exit()

    # Extract calibration from the first valid filename
    calibration = None
    for cine_filename in cine_files:
        match = re.search(r'P(\d{2})', cine_filename)
        if match:
            x = int(match.group(1))  # Extracted two-digit number as integer
            calibration = 0.004083 * x - 0.066285  # Compute calibration factor
            print(f"Extracted x: {x}, Calibration factor: {calibration:.5f}")
            break

    if calibration is None:
        print("Could not extract calibration from any file in the folder.")
        exit()

    # Use the first cine file for chamber detection
    first_cine_path = os.path.join(input_folder, cine_filename)
    first_avi_path = os.path.join(output_root, f"{cine_filename}.avi")

    try:
        # Read first CINE file
        time_arr, frame_arr = read_cine(first_cine_path)
        # Convert to AVI
        convert_cine_to_avi(frame_arr, first_avi_path)

        # Open AVI file for chamber detection
        cap = cv2.VideoCapture(first_avi_path)
        ret, initial_frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read first frame for chamber detection in {cine_filename}")
        
        PIXELS_PER_METER = 100 / calibration  # Assuming calibration is in cm/pixel
        (cx, cy), chamber_radius = detect_chamber(initial_frame, PIXELS_PER_METER)

        # Save chamber visualization
        chamber_vis = initial_frame.copy()
        cv2.circle(chamber_vis, (cx, cy), chamber_radius, (0, 255, 0), 2)
        chamber_output_path = os.path.join(output_root, "chamber_detection.png")
        cv2.imwrite(chamber_output_path, chamber_vis)
        print(f"Chamber detection saved to {chamber_output_path}")

    except Exception as e:
        print(f"Error during chamber detection: {str(e)}")
        exit()

    # Process each file using the detected chamber parameters
    for cine_filename in cine_files:
        cine_path = os.path.join(input_folder, cine_filename)
        ori_name = os.path.splitext(cine_filename)[0]  # Remove extension for naming
        
        # Create a dedicated output folder for each file
        output_folder = os.path.join(output_root, ori_name)
        os.makedirs(output_folder, exist_ok=True)

        avi_path = os.path.join(output_folder, f"{ori_name}.avi")
        output_prefix = os.path.join(output_folder, f"{ori_name}_tracking_results")

        try:
            # Read CINE file
            time_arr, frame_arr = read_cine(cine_path)
            # Convert to AVI
            convert_cine_to_avi(frame_arr, avi_path)

            # Track object using extracted calibration & chamber parameters
            track_object(avi_path, output_prefix, time_arr, calibration, cx, cy, chamber_radius)

        except Exception as e:
            print(f"Error processing {cine_filename}: {str(e)}")
