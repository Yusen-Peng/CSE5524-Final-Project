import numpy as np
from skimage import io, color
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from skimage.feature import corner_shi_tomasi, corner_peaks
from utils.tracking_utils import save_prediction, compute_gradients, build_image_pyramid, parse_args, evaluate, partial_evaluate

class KLTBoxTracking:
    """
        KLT Box Tracking class that implements the KLT tracking algorithm for bounding boxes.
    """
    def __init__(self, 
                 frames_dir: str, 
                 bounding_box_path: str, 
                 window_size=31, 
                 max_iters=100, 
                 epsilon=1e-3, 
                 pyramid_levels=3,
                 reinitialize_interval=10):
        self.frames_dir = frames_dir
        self.window_size = window_size
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.pyramid_levels = pyramid_levels
        self.reinitialize_interval = reinitialize_interval

        with open(bounding_box_path, "r") as f:
            self.box_lines = f.readlines()
        self.num_frames = len(self.box_lines)
        
        # initialize bounding box
        self.init_bounding_box = self.initialize_model()
        self.current_box = self.init_bounding_box
        self.frame_count = 0


    def initialize_keypoints(self, bbox):
        """
            use Shi-Tomasi corner detection to initialize keypoints.
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        if x2 <= x1 or y2 <= y1:
            # fallback to small dummy box in the center
            return np.array([[x1 + 1, y1 + 1]])

        frame_path = os.path.join(self.frames_dir, f"img{1000 + self.frame_count + 1:03d}.jpg")
        frame = io.imread(frame_path)
        gray = color.rgb2gray(frame)

        # Clamp to image bounds
        H, W = gray.shape
        x1 = max(0, min(x1, W - 2))
        y1 = max(0, min(y1, H - 2))
        x2 = max(x1 + 1, min(x2, W - 1))
        y2 = max(y1 + 1, min(y2, H - 1))

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return np.array([[x1 + 1, y1 + 1]])

        corners = corner_peaks(corner_shi_tomasi(roi), min_distance=3, threshold_rel=0.01)
        keypoints = np.array([[x1 + x, y1 + y] for y, x in corners])

        # Fallback if no corners found
        if len(keypoints) < 10:
            keypoints = np.array([[x, y] for x in np.linspace(x1, x2, 5) for y in np.linspace(y1, y2, 5)])
        return keypoints

    def initialize_model(self):
        """
            initialize bounding box from the first frame.
        """
        
        first_frame_path = os.path.join(self.frames_dir, "img1001.jpg")
        first_frame = io.imread(first_frame_path)
        first_frame = color.rgb2gray(first_frame)

        # parse bounding box: x1, y1 (top-left), x2, y2 (bottom-right)
        x1, y1, x2, y2 = map(int, self.box_lines[0].strip().split(","))
        self.window_width = x2 - x1
        self.window_height = y2 - y1
        return [x1, y1, x2, y2]

    def lucas_kanade_step(self, I1: np.ndarray, I2: np.ndarray, keypoints):
        """
            Perform one step of Lucas-Kanade tracking for keypoints.
        """

        # Compute gradients
        Ix, Iy = compute_gradients(I1)
        It = I2 - I1
        updated_keypoints = []
        half_w = self.window_size // 2

        for kp in keypoints:
            x, y = kp
            x0, y0 = int(x), int(y)

            # check if keypoint is within bounds
            if (x0 - half_w < 0 or x0 + half_w >= I1.shape[1] or
                y0 - half_w < 0 or y0 + half_w >= I1.shape[0]):
                updated_keypoints.append([x, y])
                continue

            # extract local window
            Ix_win = Ix[y0-half_w:y0+half_w+1, x0-half_w:x0+half_w+1].flatten()
            Iy_win = Iy[y0-half_w:y0+half_w+1, x0-half_w:x0+half_w+1].flatten()
            It_win = It[y0-half_w:y0+half_w+1, x0-half_w:x0+half_w+1].flatten()

            # Solve for motion vector
            A = np.vstack((Ix_win, Iy_win)).T
            b = -It_win
            if np.linalg.cond(A) < 1e6:
                nu, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                x += nu[0]
                y += nu[1]
                if np.linalg.norm(nu) < self.epsilon:
                    updated_keypoints.append([x, y])
                else:
                    updated_keypoints.append([x, y])
            else:
                updated_keypoints.append([x, y])

        return np.array(updated_keypoints)    

    def track_frame(self, prev_frame, next_frame, prev_bbox, frame_index, verbose=True):
        """
            track the bounding box from prev_frame to next_frame.
        """
        self.frame_count += 1
        prev_gray = color.rgb2gray(prev_frame)
        next_gray = color.rgb2gray(next_frame)

        # build image pyramids
        pyramid1 = build_image_pyramid(prev_gray, self.pyramid_levels)
        pyramid2 = build_image_pyramid(next_gray, self.pyramid_levels)

        # reinitialize keypoints periodically to prevent drift
        if self.frame_count % self.reinitialize_interval == 0:
            keypoints = self.initialize_keypoints(prev_bbox)
        else:
            keypoints = self.initialize_keypoints(prev_bbox)

        # track across pyramid levels
        for lvl in reversed(range(self.pyramid_levels)):
            scale = 2 ** lvl
            keypoints_scaled = keypoints / scale
            keypoints_scaled = self.lucas_kanade_step(pyramid1[lvl], pyramid2[lvl], keypoints_scaled)
            keypoints = keypoints_scaled * scale

        if keypoints.size:
            # filter outliers using median absolute deviation
            median = np.median(keypoints, axis=0)
            mad = np.median(np.abs(keypoints - median), axis=0)
            valid = np.all(np.abs(keypoints - median) < 3 * mad, axis=1)
            valid_keypoints = keypoints[valid]

            if valid_keypoints.size:
                cx, cy = np.mean(valid_keypoints, axis=0)
                new_x1 = int(round(cx - self.window_width / 2))
                new_y1 = int(round(cy - self.window_height / 2))
                new_x2 = new_x1 + self.window_width
                new_y2 = new_y1 + self.window_height
                
                # we still reject degenerate boxes
                if new_x2 <= new_x1 or new_y2 <= new_y1:
                    new_x1, new_y1, new_x2, new_y2 = prev_bbox
            else:
                new_x1, new_y1, new_x2, new_y2 = prev_bbox
        else:
            new_x1, new_y1, new_x2, new_y2 = prev_bbox

        # constrain motion to prevent large jumps
        max_displacement = 50
        dx = abs(new_x1 - prev_bbox[0])
        dy = abs(new_y1 - prev_bbox[1])
        if dx > max_displacement or dy > max_displacement:
            new_x1, new_y1, new_x2, new_y2 = prev_bbox

        new_bbox = [new_x1, new_y1, new_x2, new_y2]
        self.current_box = new_bbox

        if verbose:
            self.visualize_prediction(next_frame, frame_index)

        return new_bbox

    def visualize_prediction(self, image: np.ndarray, frame_index: int):
        H, W = image.shape[:2]
        image_vis = image.copy().astype(np.uint8)
        x1, y1, x2, y2 = map(int, self.current_box)

        # ensure box is within bounds
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(x1 + 1, min(x2, W - 1))
        y2 = max(y1 + 1, min(y2, H - 1))

        # draw bounding box
        GREEN = [0, 255, 0]
        image_vis[y1:y2, x1, :] = GREEN
        image_vis[y1:y2, x2-1, :] = GREEN
        image_vis[y1, x1:x2, :] = GREEN
        image_vis[y2-1, x1:x2, :] = GREEN

        os.makedirs("figures/KLT_tracking", exist_ok=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(image_vis)
        plt.axis("off")
        plt.title(f"KLT Tracking - Frame {frame_index}")
        plt.savefig(f"figures/KLT_tracking/predicted_frame_{frame_index}.jpg")
        plt.close()
        

def run_tracking():
    args = parse_args()
    bounding_box_path = args.bounding_box_path
    frames_dir = "dataset/frames"
    output_path = "results/klt_baseline_predictions.txt" if bounding_box_path.startswith("dataset") else "results/klt_own_predictions.txt"
    
    tracker = KLTBoxTracking(frames_dir=frames_dir, bounding_box_path=bounding_box_path)
    prev_frame = io.imread(os.path.join(frames_dir, "img1001.jpg"))
    

    prev_bbox = tracker.init_bounding_box
    save_prediction(1, prev_bbox[1], prev_bbox[0], tracker.window_height, tracker.window_width, output_path)

    t1 = time.time()
    for frame_index in tqdm(range(2, tracker.num_frames + 1), desc="KLT Tracking"):
        curr_frame_path = os.path.join(frames_dir, f"img1{frame_index:03d}.jpg")
        curr_frame = io.imread(curr_frame_path)
       
        curr_bbox = tracker.track_frame(prev_frame, curr_frame, prev_bbox, frame_index)
        save_prediction(frame_index, curr_bbox[1], curr_bbox[0], tracker.window_height, tracker.window_width, output_path)
        prev_frame = curr_frame
        prev_bbox = curr_bbox

    t2 = time.time()
    return (t2 - t1) / max(1, tracker.num_frames - 1)

def main():
    print("Starting KLT tracking...")
    average_elapsed_time = run_tracking()
    print("KLT tracking completed successfully.")

    print("Evaluating tracking performance...")
    gt_path = "dataset/groundtruth.txt"
    prediction_path = "results/klt_baseline_predictions.txt"
    avg_iou_20 = partial_evaluate(gt_csv_path=gt_path, prediction_csv_path=prediction_path, max_frame=20)
    avg_iou_70 = partial_evaluate(gt_csv_path=gt_path, prediction_csv_path=prediction_path, max_frame=70)
    avg_iou = evaluate(gt_csv_path=gt_path, prediction_csv_path=prediction_path)
    print(f"Average IoU (first 20 frames): {avg_iou_20:.4f}")
    print(f"Average IoU (first 70 frames): {avg_iou_70:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Time per Frame: {average_elapsed_time:.5f} seconds")
    

if __name__ == "__main__":
    main()
    print("KLT tracking script executed successfully.")