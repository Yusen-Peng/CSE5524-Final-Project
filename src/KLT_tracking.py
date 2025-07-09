import numpy as np
from skimage import io, color, transform, filters
import os
from tqdm import tqdm
from utils.tracking_utils import save_prediction, compute_gradients, build_image_pyramid, bbox_to_corners, corners_to_bbox


class KLTBoxTracking:
    """
        KLT Box Tracking class that implements the KLT tracking algorithm for bounding boxes.
    """
    def __init__(self, 
            frames_dir: str, 
            bounding_box_path: str, 
            window_size=5, 
            max_iters=10, 
            epsilon=1e-3, 
            pyramid_levels=3
        ):

        with open(bounding_box_path, "r") as f:
            box_lines = f.readlines()
        self.frames_dir = frames_dir
        self.num_frames = len(box_lines)

        # the window size for computing local gradients
        self.window_size = window_size

        # iterate (within a pyramid level) to receive a more accurate optical flow vector
        self.max_iters = max_iters # the upper bound
        self.epsilon = epsilon # the convergence threshold
        self.pyramid_levels = pyramid_levels

        # initialize from the first frame
        self.init_bounding_box = self.initialize_model(box_lines)

    def initialize_model(self, box_lines: list[str]):
        first_frame = io.imread(os.path.join(self.frames_dir, "img1001.jpg"))
        first_frame = color.rgb2gray(first_frame)

        x1, y1, x2, y2 = map(int, box_lines[0].strip().split(","))
        self.window_width = x2 - x1
        self.window_height = y2 - y1

        return [x1, y1, x2, y2]


    def lucas_kanade_step(self, I1: np.ndarray, I2: np.ndarray, keypoints):
        # compute the gradients
        Ix, Iy = compute_gradients(I1)
        It = I2 - I1

        updated_keypoints = []
        half_w = self.window_size // 2

        # track each keypoint
        for kp in keypoints:
            x, y = kp

            # iterative refinement
            for _ in range(self.max_iters):
                x0, y0 = int(x), int(y)

                # check if the keypoint is within the image bounds
                if (
                    x0 - half_w < 0 or x0 + half_w >= I1.shape[1] or
                    y0 - half_w < 0 or y0 + half_w >= I1.shape[0]
                ):
                    break

                # extract the local window (patch)
                Ix_win = Ix[y0-half_w:y0+half_w+1, x0-half_w:x0+half_w+1].flatten()
                Iy_win = Iy[y0-half_w:y0+half_w+1, x0-half_w:x0+half_w+1].flatten()
                It_win = It[y0-half_w:y0+half_w+1, x0-half_w:x0+half_w+1].flatten()

                # solve the motion vector
                A = np.vstack((Ix_win, Iy_win)).T
                b = -It_win
                nu, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                x += nu[0]
                y += nu[1]

                # check for convergence
                if np.linalg.norm(nu) < self.epsilon:
                    break

            updated_keypoints.append([x, y])
        return np.array(updated_keypoints)


    def track_frame(self, prev_frame, next_frame, prev_bbox, frame_index, verbose=True):
        prev_gray = color.rgb2gray(prev_frame)
        next_gray = color.rgb2gray(next_frame)

        # build pyramids
        pyramid1 = build_image_pyramid(prev_gray, self.pyramid_levels)
        pyramid2 = build_image_pyramid(next_gray, self.pyramid_levels)

        # identify the corners of the previous bounding box as keypoints
        corners = bbox_to_corners(prev_bbox)

        
        corners = corners * (1.0 / 2 ** (self.pyramid_levels - 1))

        for lvl in range(self.pyramid_levels):
            I1_lvl = pyramid1[lvl]
            I2_lvl = pyramid2[lvl]
            corners = self.lucas_kanade_step(I1_lvl, I2_lvl, corners)
            if lvl < self.pyramid_levels - 1:
                corners *= 2

        new_bbox = corners_to_bbox(corners)

        if verbose:
            print(f"Frame {frame_index}: Tracked bbox = {new_bbox}")

        return new_bbox





frames_dir = "dataset/frames"
bbox_path = "dataset/groundtruth.txt"
output_path = "results/klt_predictions.txt"

tracker = KLTBoxTracking(frames_dir, bbox_path)
prev_frame = io.imread(os.path.join(frames_dir, "img1001.jpg"))
prev_bbox = tracker.init_bounding_box

for frame_index in tqdm(range(2, tracker.num_frames + 1)):
    curr_frame_path = os.path.join(frames_dir, f"img1{frame_index:03d}.jpg")
    curr_frame = io.imread(curr_frame_path)

    curr_bbox = tracker.track_frame(prev_frame, curr_frame, prev_bbox, frame_index)
    save_prediction(frame_index, curr_bbox[1], curr_bbox[0], tracker.window_height, tracker.window_width, output_path)

    prev_frame = curr_frame
    prev_bbox = curr_bbox