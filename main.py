import os
import cv2
import csv
import numpy as np
import pandas as pd
from cal_refraction import refraction
from CCTV_calibration import calibration
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import datetime


start_time = datetime.datetime.now()

"""
Prepare data:
1. Camera points are used to estimate the parameter of camera with a rectangle, i.e., camera calibration,
the four points is the pixel coordinates of the four conner of the rectangle.
2. Known Length is an edge of the rectangle for camera calibration.
3. Length Direction is the use to describe the relationship between the known length and the road direction. 
There are two options, parallel and vertical.
4. Mask Path is the path of the mask image used as a benchmark, normally the first frame of non-flood image is used.
5. Object Path is the path of the image waiting for water-level measurement.
6. Save Path is the path of output.
"""

data = {'Name': ["CCTV1-22.8.25", "Simulation"],
        'Camera points': [[(286, 470), (259, 509), (418, 496), (402, 540)], [(571, 405), (458, 490), (811, 451), (735, 554)]],
        'Camara rectangular': [[(188, 443), (451, 573)], [(681, 308), (880, 596)]],
        'Known Length': [2, 3],
        'Length Direction': ["parallel", "parallel"],
        'Mask Path': ["./Example_data/image/CCTV1/2022.8.25/2022-08-25 05-13-30.jpg",
                      "./Example_data/image/Simulation/road 0 with WD_0.0m.jpg"],
        'Object Path': ["./Example_data/image/CCTV1/2022.8.25/",
                        "./Example_data/image/Simulation/"],
        'Save Path': ["./Example_data/save/CCTV1/2022.8.25/",
                      "./Example_data/save/Simulation/"],
        }
# Note that the coordinates of seed above are (vertical, horizontal)


df = pd.DataFrame(data)

for selected_index in range(2):
    if __name__ == "__main__":
        mask_image = df.loc[selected_index, 'Mask Path']
        mask = cv2.imread(mask_image)
        height, width = mask.shape[:2]
        wl_all = []
        """
        Step 1: Zoom in on the study area and select four points to calibrate the camera.===============================
        """
        # Read directly from 'Camera points' and 'Camara rectangular' in df to get rect and selected_points
        rect = df.loc[selected_index, "Camara rectangular"]
        selected_points = df.loc[selected_index, "Camera points"]

        """
        Step 2:camera calibration=======================================================================================
        """
        # Read omiga directly from 'Known Length' in df and input_object from 'Length Direction' in df. calibration
        # function is applied as usual
        omiga = df.loc[selected_index, "Known Length"]
        input_object = df.loc[selected_index, "Length Direction"]
        s, t, p, f, l, x_cam, y_cam, z_cam = calibration(height, width, float(omiga), selected_points, input_object)
        print("Camera calibration has been completed!")

        """
        Step 3: Match the points with lightglue to get matched_points1 and matched_points2.=============================
        """
        # Picture Catalog
        object_path = df.loc[selected_index, 'Object Path']
        image1_dir = Path(object_path)
        # Setting the save path
        output_dir = Path(df.loc[selected_index, 'Save Path'])
        output_dir.mkdir(exist_ok=True)

        # Setting does not calculate the gradient
        torch.set_grad_enabled(False)
        # List for storing coordinates
        data = []
        distance_data = []  # For storing distance data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Creating Feature Extractors and Matchers
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        matcher = LightGlue(features="superpoint").eval().to(device)
        top_left = (rect[0][1], rect[0][0])  # upper-left coordinate
        bottom_right = (rect[1][1], rect[1][0])  # lower-right coordinate

        # Loop to read image1 and match
        for image1_path in image1_dir.glob("*.jpg"):  # Search for .jpg files
            image1 = load_image(image1_path).cuda()

            # Cropping images
            image0 = load_image(df.loc[selected_index, 'Mask Path']).cuda()
            image0_cropped = image0[:, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
            image1_cropped = image1[:, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

            # Extraction Characteristics
            feats0 = extractor.extract(image0)
            feats1 = extractor.extract(image1)
            matches01 = matcher({"image0": feats0, "image1": feats1})

            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

            # Access to key points
            kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
            m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

            # Initialize the matchpoint list
            matched_points1 = []
            matched_points2 = []

            # Filter keypoints outside the cropping area and save valid keypoints
            for original_point, match_point in zip(m_kpts0, m_kpts1):
                x0, y0 = original_point[0].item(), original_point[1].item()
                x1, y1 = match_point[0].item(), match_point[1].item()

                # Check if the coordinates are outside the cropping area
                if not (top_left[0] <= y0 <= bottom_right[0] and top_left[1] <= x0 <= bottom_right[1]):
                    continue

                # Save original coordinates
                data.append([image1_path.name, original_point.tolist(), match_point.tolist()])

                # Calculate the Euclidean distance and save
                distance = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)  # Calculate the Euclidean distance
                distance_data.append([image1_path.name, distance])  # Add distance data

                # Save the original coordinates to the list of matching points
                matched_points1.append(original_point.tolist())  # save original_point
                matched_points2.append(match_point.tolist())  # save match_point

            # Adjust the keypoint position to fit the cropped image
            m_kpts0_cropped = m_kpts0.clone()
            m_kpts1_cropped = m_kpts1.clone()

            # Update keypoint coordinates
            m_kpts0_cropped[:, 0] -= top_left[1]
            m_kpts0_cropped[:, 1] -= top_left[0]
            m_kpts1_cropped[:, 0] -= top_left[1]
            m_kpts1_cropped[:, 1] -= top_left[0]

            # visualization
            axes = viz2d.plot_images([image0_cropped, image1_cropped])
            colors = [plt.cm.hsv(i / len(m_kpts0_cropped)) for i in range(len(m_kpts0_cropped))]
            viz2d.plot_matches(m_kpts0_cropped, m_kpts1_cropped, color=colors, lw=2)
            viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

            # Generate output image filename
            output_image_path = output_dir / f"matched_{image1_path.name}"
            plt.savefig(output_image_path)
            plt.close()  # Close the current image to save memory

            """
            Step 4: Refraction calculation==============================================================================
            """
            wl = []
            n_air = 1
            n_water = 1.334
            for idy, points in enumerate(matched_points1):
                # Coordinate system of rect: ↓y, →x, upper-left origin
                # The first point in rect is the upper left corner and the second point is the lower right corner (x,y)
                x_1 = points[0]
                y_1 = points[1]
                x_2 = matched_points2[idy][0]
                y_2 = matched_points2[idy][1]

                wl_temp = refraction(f, l, s, t, p, x_1, y_1, x_2, y_2, width, height, x_cam, y_cam, z_cam, n_air,
                                     n_water)
                if wl_temp >= -0.5:
                    # Here we set an allowable vibration offset
                    if wl_temp < 0:
                        wl_temp = 0
                    wl.append(wl_temp)
                    # Here we only save points greater than or equal to 0, because less than 0 is unlikely to happen,
                    # so the only possibility is that there was an error in the previous recognition
            # Filter it.
            if len(wl) >= 3:
                # Identifying outliers using the IQR method
                Q1 = np.percentile(wl, 25)
                Q3 = np.percentile(wl, 75)
                IQR = Q3 - Q1
                valid_iqr_indices = np.where(~((wl < Q1) | (wl > Q3)))[0]
                if valid_iqr_indices.size > 0:
                    wl_mean = np.mean([wl[index] for index in valid_iqr_indices])
                    print("After IQR eliminates outliers, " + str(valid_iqr_indices.size) + "match points are calculated.")
            else:
                wl_mean = np.mean(wl)

            wl_all.append(wl_mean)
            print("WL = ", wl_mean, "m.")

            csv_file_path = os.path.join(df.loc[selected_index, 'Save Path'], df.loc[selected_index, 'Name'] + ".csv")
            if os.path.exists(csv_file_path):
                mode = 'w'
            else:
                mode = 'x'

            # Storing wl and wl_mean into CSV files
            with open(csv_file_path, mode=mode, newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['object_image', 'wl'])
                name = os.listdir(object_path)

                for img_name, file_name_crop, wl_value in zip([df.loc[selected_index, 'Object Path']] * len(wl_all),
                                                              os.listdir(object_path), wl_all):
                    writer.writerow([img_name + file_name_crop, wl_value])

            print(image1_path, "Completed, data has been saved to a CSV file: ", csv_file_path)

end_time = datetime.datetime.now()
print (end_time - start_time)
