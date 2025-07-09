import numpy as np
from scipy.linalg import eigvals
import argparse
import csv
from skimage import io, filters
import os
from skimage import transform

def extract_features(window: np.ndarray) -> np.ndarray:
    """
        a helper function to extract features from the given window
        Features are: (X, Y, R, G, B)
    """
    H, W, _ = window.shape
    Y, X = np.mgrid[0:H, 0:W]
    X = X.flatten()
    Y = Y.flatten()
    R = window[:, :, 0].flatten()
    G = window[:, :, 1].flatten()
    B = window[:, :, 2].flatten()
    return np.stack([X, Y, R, G, B], axis=1)

def riemannian_distance(C1, C2):
    """
        Compute the Riemannian distance between two covariance matrices C1 and C2
    """
    eigenvalues = eigvals(C1, C2)
    
    # filter real, non-positive eigenvalues
    eigenvalues = np.real(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 0]

    # finally compute the distance
    EPSILON = 1e-6
    log_eigs = np.log(eigenvalues + EPSILON)
    return np.sqrt(np.sum(log_eigs**2))

def save_prediction(frame_index, best_y, best_x, window_height, window_width, output_path):
    """
        Save the predicted bounding box to the output file.
    """
    with open(output_path, "a") as f:
        xtl, ytl = best_x, best_y
        xbr, ybr = best_x + window_width, best_y + window_height
        f.write(f"{frame_index},{xtl},{ytl},{xbr},{ybr}\n")

def parse_args():
    """
        Parse command line arguments for the covariance tracking script.
    """
    parser = argparse.ArgumentParser(description="Covariance Tracking")
    parser.add_argument(
        "--bounding_box_path", 
        type=str, 
        default="dataset/groundtruth.txt", 
        help="Path to the bounding box file for the first frame."
    )
    
    return parser.parse_args()

def intersectionOverUnion(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    """
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    # when two boxes don't intersect at all
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - intersection

    return intersection / union

def evaluate(gt_csv_path, prediction_csv_path):
    """
        Compute average Intersection over Union (IoU) across all frames.
    """

    # load ground truth boxes
    gt_boxes = []
    with open(gt_csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            x1, y1, x2, y2 = map(int, row)
            gt_boxes.append((x1, y1, x2, y2))

    # load prediction boxes 
    pred_boxes = {}
    with open(prediction_csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            frame, x1, y1, x2, y2 = map(int, row)
            pred_boxes[frame] = (x1, y1, x2, y2)

    # Compute IoU metrics
    IoUs = []
    for frame_idx, gt_box in enumerate(gt_boxes):
        if frame_idx not in pred_boxes:
            print(f"Warning: Missing prediction for frame {frame_idx}")
            continue
        pred_box = pred_boxes[frame_idx]
        IoUs.append(intersectionOverUnion(gt_box, pred_box))

    # average across all frames
    return sum(IoUs) / len(IoUs)


def partial_evaluate(gt_csv_path, prediction_csv_path, max_frame=70):
    """
        Compute average IoU over the first `max_frame` frames only.
    """
    # Load ground truth boxes
    gt_boxes = []
    with open(gt_csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            x1, y1, x2, y2 = map(int, row)
            gt_boxes.append((x1, y1, x2, y2))

    # Load prediction boxes
    pred_boxes = {}
    with open(prediction_csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            frame, x1, y1, x2, y2 = map(int, row)
            pred_boxes[frame] = (x1, y1, x2, y2)

    # Compute IoU metrics for first `max_frame` frames
    IoUs = []
    for frame_idx, gt_box in enumerate(gt_boxes):
        if frame_idx >= max_frame:
            break
        if frame_idx not in pred_boxes:
            print(f"Warning: Missing prediction for frame {frame_idx}")
            continue
        pred_box = pred_boxes[frame_idx]
        IoUs.append(intersectionOverUnion(gt_box, pred_box))

    return sum(IoUs) / len(IoUs)

def circularNeighbors(
        img: np.ndarray, 
        x: float, 
        y: float, 
        radius: float
    ) -> np.ndarray:
    """
        extract a feature vector for each pixel in a circular neighborhood
    """
    height, width, _ = img.shape

    # find the bounding box
    x_range = np.arange(int(np.floor(x - radius)), int(np.ceil(x + radius))+1)
    y_range = np.arange(int(np.floor(y - radius)), int(np.ceil(y + radius))+1)

    # find all the neighbors
    X = []
    for xi in x_range:
        for yi in y_range:
            distance = np.sqrt((xi - x) ** 2 + (yi - y) ** 2)

            # check the distance: STRICTLY less than radius
            if distance < radius and 0 <= xi < width and 0 <= yi < height:
                # extract features
                R, G, B = img[yi, xi]
                X.append([xi, yi, R, G, B])

    return np.array(X)

def EpanechnikovKernel(xi, yi, x, y, h):
    """
        Epanechnikov kernel function.
    """
    r = ((xi - x) ** 2 + (yi - y) ** 2) / (h ** 2)
    if r < 1:
        return 1 - r
    else:
        return 0
    
def color2bin(R, G, B, bins):
    """
        convert RGB values to bin indices    
    """
    r_bin = int(R * bins / 256)
    g_bin = int(G * bins / 256)
    b_bin = int(B * bins / 256)

    # ensure indices are within bounds
    r_bin = min(max(r_bin, 0), bins - 1)
    g_bin = min(max(g_bin, 0), bins - 1)
    b_bin = min(max(b_bin, 0), bins - 1)

    return r_bin, g_bin, b_bin

def colorHistogram(X, bins, x, y, h):
    """
        build a color histogram from a neighborhood of points.
    """
    hist = np.zeros((bins, bins, bins), dtype=np.float32)

    for xi, yi, R, G, B in X:
        # apply the Epanechnikov kernel
        k = EpanechnikovKernel(xi, yi, x, y, h)

        # only accumulate if computed kernel is positive
        if k > 0:
            # convert RGB values to bin indices
            r_bin, g_bin, b_bin = color2bin(R, G, B, bins)

            # accumulate the histogram
            hist[r_bin, g_bin, b_bin] += k
    
    # normalize the histogram
    hist /= (np.sum(hist) + 1e-8)
    return hist

def meanshiftWeights(X, q_model, p_test, bins):
    """
        calculate a vector of the mean-shift weights - sqrt(q/p)
    """
    weights = np.zeros(X.shape[0], dtype=np.float32)

    for i, (_, _, R, G, B) in enumerate(X):
        # convert RGB values to bin indices
        r_bin, g_bin, b_bin = color2bin(R, G, B, bins)

        q = q_model[r_bin, g_bin, b_bin]
        p = p_test[r_bin, g_bin, b_bin]

        # calculate the weight
        weights[i] = np.sqrt( q / (p + 1e-8) )
    
    return weights

def compute_gradients(image) -> tuple[np.ndarray, np.ndarray]:
        Ix = filters.sobel_v(image)
        Iy = filters.sobel_h(image)
        return Ix, Iy


def build_image_pyramid(image, pyramid_levels):
    pyramid = [image]
    for _ in range(1, pyramid_levels):
        # generate each pyramid
        image = transform.rescale(image, 0.5, anti_aliasing=True)
        pyramid.append(image)
    
    # the order is from coarse to fine
    return pyramid[::-1]

def bbox_to_corners(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([
        [x1, y1],
        [x2, y1],
        [x1, y2],
        [x2, y2]
    ])

def corners_to_bbox(corners):
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]
    return [
        int(np.round(x_coords.min())),
        int(np.round(y_coords.min())),
        int(np.round(x_coords.max())),
        int(np.round(y_coords.max()))
    ]
