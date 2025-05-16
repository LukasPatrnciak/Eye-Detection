#   HLADANIE OBJEKTOV NA OBRAZE I
#   Autor: Bc. Lukas Patrnciak
#   AIS ID: 92320
#   E-mail: xpatrnciak@stuba.sk


# KNIZNICE
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# KONSTANTY
PADDING = 200
CLIP = 2.5
TITLE_SIZE = (8,8)

KERNEL_SIZE = (9,9)
SIGMA=1.5

EDGES_LOW_TRESHOLD = 60
EDGES_HIGH_TRESHOLD = 120

IOU_THRESHOLD = 0.75


# VSEOBECNE PREMENNE
dataset_path = "duhovky/"
image_path = dataset_path + "029/R/S1029R03.jpg"
ground_truth_path = "iris_annotation.csv"
ground_truth_data = pd.read_csv(ground_truth_path)

circle_params = {
    "pupil":   {"dp": 2.2, "min_dist": 5, "param1": 100, "param2": 50, "min_radius": 40, "max_radius": 50},
    "iris":    {"dp": 2.2, "min_dist": 5, "param1": 100, "param2": 30, "min_radius": 95, "max_radius": 120},
    "upper_eyelid": {"dp": 2.5, "min_dist": 100, "param1": 100, "param2": 50, "min_radius": 230, "max_radius": 290},
    "lower_eyelid": {"dp": 2.5, "min_dist": 100, "param1": 100, "param2": 50, "min_radius": 230, "max_radius": 310},
}

grid_search_ranges = {
    "pupil": {"min_radius": (20, 60, 5), "max_radius": (50, 80, 5)},
    "iris": {"min_radius": (70, 120, 5), "max_radius": (120, 170, 5)},
    "upper_eyelid": {"min_radius": (180, 250, 20), "max_radius": (240, 360, 20)},
    "lower_eyelid": {"min_radius": (180, 270, 20), "max_radius": (230, 400, 20)},
}

circle_colors = {
    "pupil": (0, 0, 255),
    "iris": (0, 255, 0),
    "upper_eyelid": (255, 0, 0),
    "lower_eyelid": (255, 255, 0)
}

ground_truth = {
    "pupil": (160 + PADDING, 156 + PADDING, 49),
    "iris": (158 + PADDING, 158 + PADDING, 107),
    "upper_eyelid": (183 + PADDING, -49 + PADDING, 281),
    "lower_eyelid": (200 + PADDING, 417 + PADDING, 328)
}

ground_truths = {}

for index, row in ground_truth_data.iterrows():
    dataframe_image = row['image']

    ground_truths[dataframe_image] = {
        "pupil": (row['center_x_1'] + PADDING, row['center_y_1'] + PADDING, row['polomer_1']),
        "iris": (row['center_x_2'] + PADDING, row['center_y_2'] + PADDING, row['polomer_2']),
        "upper_eyelid": (row['center_x_3'] + PADDING, row['center_y_3'] + PADDING, row['polomer_3']),
        "lower_eyelid": (row['center_x_4'] + PADDING, row['center_y_4'] + PADDING, row['polomer_4'])
    }


# FUNKCIE
def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def enhance_contrast(image, clip_limit=CLIP,  tile_size=TITLE_SIZE):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    return clahe.apply(image)

def denoise_image(image, kernel_size=KERNEL_SIZE, sigma=SIGMA):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def detect_edges(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

def detect_circles(image, dp, min_dist, param1, param2, min_radius, max_radius):
    return cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp, min_dist,
                            param1=param1, param2=param2,
                            minRadius=min_radius, maxRadius=max_radius)

def iou(A, B):
    xA = A[0]
    yA = A[1]
    rA = A[2]

    xB = B[0]
    yB = B[1]
    rB = B[2]

    d = np.sqrt((xB - xA) ** 2 + (yB - yA) ** 2)

    areaA = np.pi * (rA ** 2)
    areaB = np.pi * (rB ** 2)

    if d >= rA + rB:
        return 0.0

    if d <= abs(rA - rB):
        intersection_area = np.pi * min(rA, rB) ** 2

        return intersection_area / areaA if areaA > areaB else intersection_area / areaB

    term1 = rA ** 2 * np.arccos((d ** 2 + rA ** 2 - rB ** 2) / (2 * d * rA))
    term2 = rB ** 2 * np.arccos((d ** 2 + rB ** 2 - rA ** 2) / (2 * d * rB))
    term3 = 0.5 * np.sqrt((-d + rA + rB) * (d + rA - rB) * (d - rA + rB) * (d + rA + rB))

    intersection_area = term1 + term2 - term3
    union_area = areaA + areaB - intersection_area

    return intersection_area / union_area

def evaluate_detection(dcs, gt, iou_threshold=IOU_THRESHOLD):
    tp, fp, fn = 0, 0, 0
    metrics_dict = {}
    circle_iou, precision, recall, f1_score = 0, 0, 0, 0

    for name, gt_circle in gt.items():
        if name in dcs:
            detected_circle = dcs[name]
            circle_iou = iou(detected_circle, gt_circle)

            if circle_iou >= iou_threshold:
                tp += 1

            else:
                fp += 1
        else:
            fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics_dict[name] = (circle_iou, precision, recall, f1_score)

    return metrics_dict

def filter_circles(detected_circles, img_shape, tolerance):
    h, w = img_shape
    cx, cy = w // 2, h // 2
    tol_x, tol_y = tolerance * w, tolerance * h

    best_circles = {}
    iris_center_y = None

    if "iris" in detected_circles and detected_circles["iris"]:
        iris_center_y = np.mean([c[1] for c in detected_circles["iris"]])

    for feature, circles in detected_circles.items():
        if not circles:
            continue

        if feature == "pupil":
            min_distance = float("inf")
            best_circle = circles[0]

            for c in circles:
                x = c[0]
                y = c[1]

                if (cx - tol_x <= x <= cx + tol_x) and (cy - tol_y <= y <= cy + tol_y):
                    distance = abs(x - cx) + abs(y - cy)

                    if distance < min_distance:
                        min_distance = distance
                        best_circle = c

            for i in range(len(detected_circles)):
                if best_circle[2] >= detected_circles['iris'][i][2]:
                    continue
                else:
                    best_circles[feature] = best_circle if best_circle else circles[0]

        elif feature == "iris":
            min_distance = float("inf")
            best_circle = circles[0]

            for c in circles:
                x = c[0]
                y = c[1]

                if (cx - tol_x <= x <= cx + tol_x) and (cy - tol_y <= y <= cy + tol_y):
                    distance = abs(x - cx) + abs(y - cy)

                    if distance < min_distance:
                        min_distance = distance
                        best_circle = c

            best_circles[feature] = best_circle if best_circle else circles[0]

        elif feature == "upper_eyelid" and iris_center_y is not None:
            max_y = min(c[1] for c in circles)
            target_y = (iris_center_y + max_y) / 2

            best_circle = min(circles, key=lambda c: abs(c[1] - target_y))
            best_circles[feature] = best_circle

        elif feature == "lower_eyelid" and iris_center_y is not None:
            min_y = max(c[1] for c in circles)
            target_y = (iris_center_y + min_y) / 2

            best_circle = min(circles, key=lambda c: abs(c[1] - target_y))
            best_circles[feature] = best_circle

    return best_circles

def segment_iris(image, iris_circle, pupil_circle, upper_eyelid_circle, lower_eyelid_circle):
    mask = np.zeros_like(image, dtype=np.uint8)

    if iris_circle and pupil_circle and upper_eyelid_circle and lower_eyelid_circle is not None:
        x_iris, y_iris, r_iris = iris_circle
        x_pupil, y_pupil, r_pupil = pupil_circle
        x_upper, y_upper, r_upper = upper_eyelid_circle
        x_lower, y_lower, r_lower = lower_eyelid_circle

        cv2.circle(mask, (x_iris, y_iris), r_iris, (1,1,1), thickness=-1)

        cv2.circle(mask, (x_pupil, y_pupil), r_pupil, (0,0,0), thickness=-1)

        upper_mask = np.zeros_like(image, dtype=np.uint8)
        lower_mask = np.zeros_like(image, dtype=np.uint8)

        cv2.circle(upper_mask, (x_upper, y_upper), r_upper, (1,1,1), thickness=-1)
        cv2.circle(lower_mask, (x_lower, y_lower), r_lower, (1,1,1), thickness=-1)

        mask = mask * upper_mask * lower_mask

    return np.where(mask == 1, 1, 0)

def grid_search(image, edges, ranges):
    detected_circles = {}

    for feature, feature_ranges in ranges.items():
        min_range = feature_ranges["min_radius"]
        max_range = feature_ranges["max_radius"]

        for min_radius in range(min_range[0], min_range[1] + 1, min_range[2]):
            for max_radius in range(max_range[0], max_range[1] + 1, max_range[2]):

                params = {"dp": circle_params[feature]["dp"], "min_dist": circle_params[feature]["min_dist"], "param1": circle_params[feature]["param1"], "param2": circle_params[feature]["param2"],
                          "min_radius": min_radius, "max_radius": max_radius}

                detected = detect_circles(edges, **params)

                if detected is not None:
                    circles = []

                    for one_circle in np.uint16(np.around(detected[0])):
                        x, y, r = one_circle
                        circles.append((x, y, r, params))

                    detected_circles[feature] = circles
                else:
                    continue

    selected_circles = filter_circles(detected_circles, image.shape, 0.7)

    output_circles = {'pupil': (selected_circles['pupil'][0], selected_circles['pupil'][1], selected_circles['pupil'][2]),
                      'iris': (selected_circles['iris'][0], selected_circles['iris'][1], selected_circles['iris'][2]),
                      'lower_eyelid': ( selected_circles['lower_eyelid'][0], selected_circles['lower_eyelid'][1], selected_circles['lower_eyelid'][2]),
                      'upper_eyelid': (selected_circles['upper_eyelid'][0], selected_circles['upper_eyelid'][1], selected_circles['upper_eyelid'][2]),
                      }
    output_params = {'pupil': selected_circles['pupil'][3],
                     'iris': selected_circles['iris'][3],
                     'lower_eyelid': selected_circles['lower_eyelid'][3],
                     'upper_eyelid': selected_circles['upper_eyelid'][3]}

    return output_circles, output_params


# SPRACOVANIE OBRAZKA
img = load_image(image_path)
img = enhance_contrast(img)
img = denoise_image(img)
img = cv2.copyMakeBorder(img, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT)

circle_edges = detect_edges(img, EDGES_LOW_TRESHOLD, EDGES_HIGH_TRESHOLD)


# DETEKCIA A VYKRESLENIE JEDNOTLIVYCH KRUHOV
detected_all_circles = {}

for circle_name, circle_selected_params in circle_params.items():
    circle_detected_circle = detect_circles(circle_edges, **circle_selected_params)

    if circle_detected_circle is not None:
        detected_all_circles[circle_name] = []

        for circle in np.uint16(np.around(circle_detected_circle[0])):
            x_, y_, r_ = circle
            detected_all_circles[circle_name].append((x_, y_, r_))


# FILTROVANIE NAJLSPESICH KRUZNIC
filtered_circles = filter_circles(detected_all_circles, img.shape, 0.0)


# VYPISANIE VYSLEDKOV
circle_metrics_dict = evaluate_detection(filtered_circles, ground_truth, iou_threshold=0.75)

print(f"Detected Circles: {detected_all_circles}")
print(f"Best Circles: {filtered_circles}")
print(f"Metrics: {circle_metrics_dict}")


# VYKRESLENIE OBRAZKOV
iris = filtered_circles['iris']
pupil = filtered_circles['pupil']
upper_eyelid = filtered_circles['upper_eyelid']
lower_eyelid = filtered_circles['lower_eyelid']

segmented_iris = segment_iris(img, iris, pupil, upper_eyelid, lower_eyelid)

axs = plt.subplots(1, 2, figsize=(12, 6))[1]
axs[0].imshow(img, cmap='gray')
axs[0].set_title("Original Picture")

axs[1].imshow(circle_edges, cmap='gray')
axs[1].set_title("Detected Edges")
plt.show()

filtered_circles_output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for circle_feature, (x_, y_, r_) in filtered_circles.items():
    cv2.circle(filtered_circles_output, (x_, y_), r_, circle_colors[circle_feature], 2)
    cv2.circle(filtered_circles_output, (x_, y_), 1, circle_colors[circle_feature], 3)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(cv2.cvtColor(filtered_circles_output, cv2.COLOR_BGR2RGB))
axs[0].set_title("Filtered Circles")

axs[1].imshow(segmented_iris, cmap='gray')
axs[1].set_title("Iris Segmentation")
plt.show()


# GRID SEARCH
results = {}

for person in os.listdir(dataset_path):
    best_f1_score = 0

    for eye in ["L", "R"]:
        image_folder = os.path.join(dataset_path, person, eye)

        for image_file in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_file)

            relative_image_path = os.path.relpath(image_path, dataset_path)
            final_image_path = relative_image_path.replace(os.sep, '/')

            img = load_image(image_path)
            img = enhance_contrast(img)
            img = denoise_image(img)
            computed_edges = detect_edges(img, EDGES_LOW_TRESHOLD, EDGES_HIGH_TRESHOLD)

            computed_output_circles, computed_output_params = grid_search(img, computed_edges, grid_search_ranges)
            circle_metrics_dict = evaluate_detection(computed_output_circles, ground_truths[final_image_path], iou_threshold=0.75)

            f_1_pupil = circle_metrics_dict['pupil'][3]
            f_1_iris = circle_metrics_dict['iris'][3]
            f_1_upper_eyelid = circle_metrics_dict['upper_eyelid'][3]
            f_1_lower_eyelid = circle_metrics_dict['lower_eyelid'][3]

            average_f1_score = (f_1_pupil + f_1_iris + f_1_upper_eyelid + f_1_lower_eyelid) / 4

            if average_f1_score > best_f1_score:
                results.clear()
                best_f1_score = average_f1_score
                results[final_image_path] = (computed_output_params, computed_output_circles, circle_metrics_dict)

print("RESULTS: ", results)