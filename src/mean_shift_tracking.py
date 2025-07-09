from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as patches
import os
import time
from utils.tracking_utils import circularNeighbors, colorHistogram, meanshiftWeights, parse_args, save_prediction, evaluate, partial_evaluate

class MeanShiftTracking:
    """
    Mean Shift Tracking class that implements the mean shift tracking algorithm.
    Args:
        frames_dir (str): Directory containing the frames.
        bounding_box_path (str): Path to the bounding box file for the first frame.
        bins (int): Number of bins for the color histogram.
        h (int): Radius of the circular neighborhood.
    """
    def __init__(self, frames_dir: str, bounding_box_path: str, bins: int = 16, h: int = 25):
        self.bins = bins
        self.h = h

        with open(bounding_box_path, "r") as f:
            self.box_lines = f.readlines()
        self.num_frames = len(self.box_lines)

        first_frame = io.imread(os.path.join(frames_dir, "img1001.jpg")).astype(np.float32)
        x1, y1, x2, y2 = map(int, self.box_lines[0].strip().split(","))

        # compute the initial center
        x0 = (x1 + x2) / 2
        y0 = (y1 + y2) / 2

        self.window_width = x2 - x1
        self.window_height = y2 - y1

        # initialize the current position
        self.current_x = x0
        self.current_y = y0

        # intialize the model
        X_model = circularNeighbors(first_frame, x0, y0, h)
        self.q_model = colorHistogram(X_model, bins, x0, y0, h)

    def mean_shift_step(self, image: np.ndarray, num_iterations: int = 25) -> tuple:
        """
            perform the mean shift step for the given image.
        """
        for _ in range(num_iterations):
            X_target = circularNeighbors(image, self.current_x, self.current_y, self.h)
            p_test = colorHistogram(X_target, self.bins, self.current_x, self.current_y, self.h)
            weights = meanshiftWeights(X_target, self.q_model, p_test, self.bins)

            next_x = np.sum(X_target[:, 0] * weights) / (np.sum(weights) + 1e-8)
            next_y = np.sum(X_target[:, 1] * weights) / (np.sum(weights) + 1e-8)

            self.current_x, self.current_y = next_x, next_y

        return self.current_y, self.current_x

    def visualize_prediction(self, image: np.ndarray, frame_index: int):
        """
            visualize the predicted bounding box on the image and save it.
        """
        H, W, _ = image.shape
        
        image_vis = image.copy().astype(np.uint8)
        
        # get back to the upper left corner of the bounding box
        x = int(self.current_x - self.window_width / 2)
        y = int(self.current_y - self.window_height / 2)

        # Ensure box is within bounds
        x = max(0, min(x, W - self.window_width))
        y = max(0, min(y, H - self.window_height))

        GREEN = [0, 255, 0]
        image_vis[y:y+self.window_height, x, :] = GREEN
        image_vis[y:y+self.window_height, x+self.window_width-1, :] = GREEN
        image_vis[y, x:x+self.window_width, :] = GREEN
        image_vis[y+self.window_height-1, x:x+self.window_width, :] = GREEN

        plt.figure(figsize=(10, 10))
        plt.imshow(image_vis)
        plt.axis("off")
        plt.title(f"Mean Shift - Frame {frame_index}")
        plt.savefig(f"figures/meanshift_tracking/prediction_frame_{frame_index}.jpg")
        plt.close()


def run_meanshift_tracking():
    args = parse_args()
    bounding_box_path = args.bounding_box_path
    frames_dir = "dataset/frames"

    if bounding_box_path.startswith("dataset"):
        prediction_output_path = "results/meanshift_baseline_predictions.txt"
    else:
        prediction_output_path = "results/meanshift_own_predictions.txt"

    tracker = MeanShiftTracking(frames_dir=frames_dir, bounding_box_path=bounding_box_path)

    t1 = time.time()
    for frame_index in tqdm(range(1, tracker.num_frames + 1), desc="MeanShift Tracking"):
        frame_path = os.path.join(frames_dir, f"img1{frame_index:03d}.jpg")
        image = io.imread(frame_path).astype(np.float32)

        best_y, best_x = tracker.mean_shift_step(image=image, num_iterations=50)
        tracker.visualize_prediction(image, frame_index)
        save_prediction(frame_index, int(best_y), int(best_x),
                        tracker.window_height, tracker.window_width,
                        prediction_output_path)
    t2 = time.time()

    return (t2 - t1) / tracker.num_frames


def main():
    print("Starting Mean-Shift tracking...")
    avg_time = run_meanshift_tracking()
    #avg_time = 12345 # Placeholder for average time calculation
    print("Tracking completed.")

    print("Evaluating tracking performance...")
    gt_path = "dataset/groundtruth.txt"
    pred_path = "results/meanshift_baseline_predictions.txt"
    avg_iou_20 = partial_evaluate(gt_csv_path=gt_path, prediction_csv_path=pred_path, max_frame=20)
    avg_iou_70 = partial_evaluate(gt_csv_path=gt_path, prediction_csv_path=pred_path, max_frame=70)
    avg_iou = evaluate(gt_csv_path=gt_path, prediction_csv_path=pred_path)
    print(f"Average IoU (first 20 frames): {avg_iou_20:.4f}")
    print(f"Average IoU (first 70 frames): {avg_iou_70:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Time per Frame: {avg_time:.5f} seconds")


if __name__ == "__main__":
    main()
