import numpy as np
from scipy.linalg import eigvals
import argparse
import csv

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
