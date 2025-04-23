import cv2 as cv

import numpy as np

import tensorflow as tf

import re

import hashlib


def custom_hash(key, num_buckets=10):
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    return hashed_key


def check_password(password):
    # Check for at least one capital letter
    capital_letter_regex = r"[A-Z]"
    if not re.search(capital_letter_regex, password):
        return False

    # Check for at least one symbol
    symbol_regex = r"[\W_]"
    if not re.search(symbol_regex, password):
        return False

    # Check for at least one digit
    digit_regex = r"\d"
    if not re.search(digit_regex, password):
        return False

    # Check for length (minimum 8 characters)
    if len(password) < 8:
        return False

    # All criteria met
    return True


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def preprocess(main_image):
    # Convert image to gray
    gray_image = cv.cvtColor(main_image, cv.COLOR_BGR2GRAY)
    rows, cols = gray_image.shape

    # Reduce noise with median blur
    median_blurred_image = cv.medianBlur(gray_image, 9)
    return median_blurred_image


def detect_circles(
    preprocessed_image, min_radius, max_radius, radius_diff, expected_radius_diff
):
    # Detect circles using Hough transform
    circles = cv.HoughCircles(
        preprocessed_image,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=preprocessed_image.shape[0] // 8,
        param1=100,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    # Check if circles are detected
    if circles is None or len(circles) != 1:
        raise ValueError("Error: Incorrect number of circles detected.")

    # Extract pupil center and radius
    pupil_center = (circles[0, 0, 0], circles[0, 0, 1])
    pupil_radius = circles[0, 0, 2]

    # Extract iris radius using intensity analysis
    rows, cols = preprocessed_image.shape
    xc, yc = pupil_center
    radius_range = range(int(pupil_radius * 1.4), int(pupil_radius * 2.4))
    intensity_sum = np.zeros(len(radius_range))

    for i, radius in enumerate(radius_range):
        for theta in range(0, 360):
            x = int(xc + radius * np.cos(np.deg2rad(theta)))
            y = int(yc - radius * np.sin(np.deg2rad(theta)))
            if 0 <= x < cols and 0 <= y < rows:
                intensity_sum[i] += preprocessed_image[y, x]

    iris_radius = min_radius  # Default value
    max_val = float("-inf")
    for i in range(2, len(intensity_sum) - 2):
        val = (
            intensity_sum[i + 2]
            + intensity_sum[i + 1]
            - intensity_sum[i - 1]
            - intensity_sum[i - 2]
        )
        if val > max_val and (radius_range[i] - pupil_radius) > radius_diff:
            max_val = val
            iris_radius = radius_range[i]

    pupil_center = tuple(map(int, pupil_center))
    pupil_radius = int(pupil_radius)
    iris_radius = int(iris_radius)

    if iris_radius - pupil_radius < expected_radius_diff:
        raise ValueError("radius differencd too small")

    return pupil_center, pupil_radius, iris_radius


def custom_sobel_filter(image, xmin, xmax, ymin, ymax):
    # Initialize Sobel matrix
    sobel_matrix = np.zeros((ymax - ymin, xmax - xmin))

    # Compute Sobel filtering using loop-based implementation
    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            # Check if indices are within bounds
            if (
                i - 1 >= 0
                and j - 1 >= 0
                and i + 1 < image.shape[0]
                and j + 1 < image.shape[1]
            ):
                sobel_matrix[i - ymin][j - xmin] = abs(
                    -image[i - 1][j - 1]
                    - 2 * image[i][j - 1]
                    - image[i + 1][j - 1]
                    + image[i - 1][j + 1]
                    + 2 * image[i][j + 1]
                    + image[i + 1][j + 1]
                )  # Transposed indexing
            else:
                sobel_matrix[i - ymin][j - xmin] = 0

    return sobel_matrix


def sobel_filter(image, pupil_center, pupil_radius, iris_radius, threshold):
    xc, yc = pupil_center
    length = pupil_radius * 2
    height = iris_radius - pupil_radius
    start_point_upper = (int(xc - length / 2), yc - iris_radius)
    end_point_upper = (int(xc + length / 2), yc - pupil_radius - 5)
    start_point_lower = (int(xc - length / 2), yc + iris_radius)
    end_point_lower = (int(xc + length / 2), yc + pupil_radius + 5)

    # Ensure coordinates are non-negative
    xmin_upper, ymin_upper = start_point_upper
    xmax_upper, ymax_upper = end_point_upper
    xmin_upper = max(0, xmin_upper)
    ymin_upper = max(0, ymin_upper)
    xmax_upper = max(0, xmax_upper)
    ymax_upper = max(0, ymax_upper)
    # For the lower region
    xmin_lower, ymax_lower = start_point_lower
    xmax_lower, ymin_lower = end_point_lower

    # Call custom_sobel_filter for upper and lower regions
    sobel_matrix_upper = custom_sobel_filter(
        image, xmin_upper, xmax_upper, ymin_upper, ymax_upper
    )
    sobel_matrix_lower = custom_sobel_filter(
        image, xmin_lower, xmax_lower, ymin_lower, ymax_lower
    )

    # Compute edge counts
    edge_counts_upper = np.sum(sobel_matrix_upper, axis=1)
    edge_counts_lower = np.sum(sobel_matrix_lower, axis=1)

    # Normalize edge counts
    total_edge_count_upper = np.sum(edge_counts_upper)
    total_edge_count_lower = np.sum(edge_counts_lower)
    edge_counts_normalized_upper = (
        edge_counts_upper / total_edge_count_upper
        if total_edge_count_upper != 0
        else np.zeros_like(edge_counts_upper)
    )
    edge_counts_normalized_lower = (
        edge_counts_lower / total_edge_count_lower
        if total_edge_count_lower != 0
        else np.zeros_like(edge_counts_lower)
    )

    # Find eyelid rows
    max_edge_upper = np.max(edge_counts_normalized_upper)
    max_edge_lower = np.max(edge_counts_normalized_lower)
    # print(f"Max Edge Lower Intensity: {max_edge_lower}, Max Edge Upper Intensity: {max_edge_upper}")

    eyelid_row_upper = (
        ymin_upper + np.argmax(edge_counts_normalized_upper)
        if max_edge_upper >= threshold
        else yc - iris_radius
    )
    eyelid_row_lower = (
        ymin_lower + np.argmax(edge_counts_normalized_lower)
        if max_edge_lower >= threshold
        else yc + iris_radius
    )

    eyelid_row_upper = max(0, eyelid_row_upper)
    eyelid_row_lower = min(eyelid_row_lower, image.shape[0] - 1)
    return eyelid_row_upper, eyelid_row_lower


def generate_circle_masks(center, radius1, radius2, shape):
    black_background = np.zeros(shape[:2], dtype=np.uint8)
    mask1 = cv.circle(black_background.copy(), center, radius1, 255, -1)
    mask2 = cv.circle(black_background.copy(), center, radius2, 255, -1)
    mask = cv.subtract(mask2, mask1)
    return mask1, mask2, mask


def apply_mask(image, mask):
    result = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    result = cv.bitwise_and(result, mask)
    return result


def extract_result_region(result, eyelid_row_upper, eyelid_row_lower, yc, iris_radius):
    if eyelid_row_upper is not None:
        result = result[eyelid_row_upper:, :]
        a = eyelid_row_upper
    else:
        result = result[yc - iris_radius :, :]
        a = yc - iris_radius
    if eyelid_row_lower:
        result = result[: eyelid_row_lower - a, :]
    else:
        result = result[: iris_radius - a, :]
    return result


def extract_iris_region(result, xc, iris_radius):
    result = result[:, xc - iris_radius : xc + iris_radius]
    xc, yc = result.shape[0] // 2, result.shape[1] // 2
    return result, xc, yc


def rubber_sheet_mapping(iris_image, r1, r2, xc, yc):
    gray = iris_image
    radial_resolution = r2 - r1
    pupil_center = (xc, yc)
    theta_steps = 360
    theta_range = np.linspace(0, 2 * np.pi, theta_steps)
    normalized_iris = np.zeros((r2 - r1, theta_steps), dtype=np.uint8)
    for r in range(radial_resolution):
        for theta_idx, theta in enumerate(theta_range):
            x_polar = pupil_center[0] + (r1 + r) * np.cos(theta)
            y_polar = pupil_center[1] + (r1 + r) * np.sin(theta)
            x_polar = max(0, min(gray.shape[1] - 1, int(x_polar)))
            y_polar = max(0, min(gray.shape[0] - 1, int(y_polar)))
            normalized_iris[r, theta_idx] = gray[int(y_polar), int(x_polar)]
    return normalized_iris


def process_image(
    main_image_backup,
    pupil_center,
    pupil_radius,
    iris_radius,
    eyelid_row_upper,
    eyelid_row_lower,
):
    hh, ww = main_image_backup.shape[:2]
    mask1, mask2, mask = generate_circle_masks(
        pupil_center, pupil_radius, iris_radius, (hh, ww)
    )
    result = apply_mask(main_image_backup, mask)
    result = extract_result_region(
        result, eyelid_row_upper, eyelid_row_lower, pupil_center[1], iris_radius
    )
    result, xc, yc = extract_iris_region(result, pupil_center[0], iris_radius)
    return result, xc, yc


def normalize_enchance_image(
    main_image_path,
    min_radius=10,
    max_radius=50,
    threshold=1,
    radius_diff=20,
    expected_radius_diff=25,
):
    if isinstance(main_image_path, str):
        main_image = cv.imread(main_image_path)
    else:
        in_memory_file = (
            main_image_path  # Assuming you have the InMemoryUploadedFile object
        )
        image_data = in_memory_file.read()  # Read the raw image bytes
        image_array = np.frombuffer(
            image_data, np.uint8
        )  # Create NumPy array from bytes
        main_image = cv.imdecode(image_array, cv.IMREAD_COLOR)
    preprocessed_image = preprocess(main_image)
    pupil_center, pupil_radius, iris_radius = detect_circles(
        preprocessed_image, min_radius, max_radius, radius_diff, expected_radius_diff
    )
    eyelid_row_upper, eyelid_row_lower = sobel_filter(
        preprocessed_image, pupil_center, pupil_radius, iris_radius, threshold=threshold
    )
    result, pupil_center_x, pupil_center_y = process_image(
        main_image,
        pupil_center,
        pupil_radius,
        iris_radius,
        eyelid_row_upper,
        eyelid_row_lower,
    )
    normalized_image = rubber_sheet_mapping(
        result, pupil_radius, iris_radius, pupil_center_x, pupil_center_y
    )
    clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    enhanced_iris = clahe.apply(normalized_image)
    enhanced_iris = cv.cvtColor(enhanced_iris, cv.COLOR_GRAY2BGR)
    enhanced_iris = np.asarray(enhanced_iris)
    # enhanced_iris = enhance_iris(enhanced_iris)
    return result, normalized_image, enhanced_iris


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_and_preprocess_image(image):

    # image = tf.io.read_file(image)
    # image = tf.image.decode_bmp(image, channels=0)
    # image=tf.image.grayscale_to_rgb(image)

    image = tf.image.resize(image, size=(224, 224))
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)
    return image
