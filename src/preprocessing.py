import cv2
import numpy as np

def preprocess_image(image_path, size=(128, 128)):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Median filter for noise removal (Paper 1 method)
    filtered = cv2.medianBlur(gray, 5)
    # Resize to uniform size
    resized = cv2.resize(filtered, size)
    # Normalize
    normalized = resized / 255.0
    return normalized, resized

def apply_canny(image):
    edges = cv2.Canny((image * 255).astype(np.uint8), 100, 200)
    return edges

def watershed_segment(image):
    img_uint8 = (image * 255).astype(np.uint8)
    _, thresh = cv2.threshold(img_uint8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    return markers