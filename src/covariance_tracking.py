import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from utils.tracking_utils import extract_features, riemannian_distance, save_prediction, parse_args, evaluate

class CovarianceTracking:
    """
        Covariance Tracking class that implements the covariance tracking algorithm.
        Args:
            frames_dir (str): Directory containing the frames.
            bounding_box_path (str): Path to the bounding box file for the first frame.
            note that we can either use the ground truth or the our predicted/extracted bounding box.
    """

    def __init__(self, frames_dir: str, bounding_box_path: str):
        
        with open(bounding_box_path, "r") as f:
            box_lines = f.readlines()
        
        # compute the total number of frames
        self.num_frames = len(box_lines)

        # extract the model covariance matrix and window size
        model_cov, window_height, window_width = self.InitializeModel(
            frames_dir=frames_dir,
            box_lines=box_lines
        )

        self.modelCovMatrix = model_cov
        self.window_height = window_height
        self.window_width = window_width
        

    def InitializeModel(self, frames_dir: str, box_lines: list[str]) -> tuple:
        """
            Initialize the object model using the first frame.
        """

        # load the bounding box from the first frame
        first_frame = io.imread(os.path.join(frames_dir, "img1001.jpg"))
        x1, y1, x2, y2 = map(int, box_lines[0].strip().split(","))
        
        # compute the window size
        window_height = y2 - y1
        window_width = x2 - x1

        # build the model covariance matrix
        model_patch = first_frame[y1:y2, x1:x2]
        model_features = extract_features(model_patch)
        model_cov = np.cov(model_features, rowvar=False, bias=True)
        return model_cov, window_height, window_width

    def build_match_distance_matrix(self, image: np.ndarray) -> np.ndarray:
        """
            Compute the match distance matrix for the given image and model covariance matrix.
        """
        
        H, W, _ = image.shape
        
        # intialize the match distance matrix
        match_distance_matrix = np.zeros(
            (H - self.window_height + 1, W - self.window_width + 1), 
            dtype=np.float32
        )

        for y in range(H - self.window_height + 1):
            for x in range(W - self.window_width + 1):
                
                # extract windows
                window = image[y:y + self.window_height, x:x + self.window_width]

                # feature extraction
                features = extract_features(window)

                # covariance matrix
                cov_mat = np.cov(features, rowvar=False, bias=True)

                # distance measure
                distance = riemannian_distance(self.modelCovMatrix, cov_mat)

                # save the match distance for each box location
                match_distance_matrix[y, x] = distance
        
        return match_distance_matrix

    def visualize_match_distance_matrix(self, match_distance_matrix: np.ndarray, frame_index: int):

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(match_distance_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Riemannian Distance', orientation='horizontal')
        plt.title('Match Distance Matrix')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(f'figures/covariance_tracking/match_distance/matrix_{frame_index}.jpg')
        plt.close(fig)

    def visualize_best_match(
            self,
            image: np.ndarray,
            frame_index: int,
            best_y: int,
            best_x: int,
        ):
        """
            visualize the best match bounding box on the original image.
        """

        # for visualization, draw bounding box on the original image
        image_with_box = image.copy().astype(np.uint8)
        rr, cc = best_y, best_x
        GREEN = [0, 255, 0]
        image_with_box[rr:rr+self.window_height, cc, :] = GREEN
        image_with_box[rr:rr+self.window_height, cc+self.window_width-1, :] =  GREEN
        image_with_box[rr, cc:cc+self.window_width, :] = GREEN
        image_with_box[rr+self.window_height-1, cc:cc+self.window_width, :] = GREEN

        # save the result image
        plt.figure(figsize=(10, 10))
        plt.imshow(image_with_box.astype(np.uint8))
        plt.title("Best Match Bounding Box")
        plt.axis('off')
        plt.savefig(f'figures/covariance_tracking/best_match_frame_{frame_index}.jpg')
        plt.close()


    def covariance_tracking(self, image: np.ndarray, frame_index: int, verbose: bool = True):
        """
            Perform covariance tracking on the given image.
        """
        # compute the match distance matrix
        match_distance_matrix = self.build_match_distance_matrix(image=image)

        if verbose:
            # visualize the match distance matrix (optional)
            print("Visualizing match distance matrix...")
            self.visualize_match_distance_matrix(match_distance_matrix, frame_index)

        # provide the location of the best candidate
        best_y, best_x = np.unravel_index(
            np.argmin(match_distance_matrix), 
            match_distance_matrix.shape
        )

        if verbose:
            # visualize the best match (optional)
            print("Visualizing best match...")
            self.visualize_best_match(
                image=image,
                frame_index=frame_index,
                best_y=best_y,
                best_x=best_x
            )

        return best_y, best_x

def run_tracking():
    
    # compare our feature extraction approach with the baseline (ground truth)
    args = parse_args()
    bounding_box_path: str = args.bounding_box_path    
    if bounding_box_path.startswith("dataset"):
        prediction_output_path = "results/baseline_predictions.txt"
    else:
        prediction_output_path = "results/own_predictions.txt"

    # run object tracking
    frames_dir = "dataset/frames"

    # timing
    t1 = time.time()

    tracker = CovarianceTracking(frames_dir=frames_dir, bounding_box_path=bounding_box_path)

    for frame_index in tqdm(range(1, tracker.num_frames + 1), desc="Tracking frames"):
        frame_path = os.path.join(frames_dir, f"img1{frame_index:03d}.jpg")
        image = io.imread(frame_path)

        best_y, best_x = tracker.covariance_tracking(
            image=image,
            frame_index=frame_index,
            verbose=True
        )

        # save the model's prediction
        save_prediction(frame_index, best_y, best_x, tracker.window_height, tracker.window_width, prediction_output_path)

    t2 = time.time()
    average_elapsed_time = (t2 - t1) / tracker.num_frames
    return average_elapsed_time

def main():
    # run the covariance tracking
    print("Starting covariance tracking...")
    average_elapsed_time = run_tracking()
    print("Covariance tracking completed successfully.")


    # evaluate the tracking performance
    print("Evaluating tracking performance...")
    gt_path = "dataset/groundtruth.txt"
    prediction_path = "results/own_predictions.txt"
    average_iou = evaluate(gt_csv_path=gt_path, prediction_csv_path=prediction_path)
    print(f"Average IoU: {average_iou:.4f}")
    print(f"Average Elapsed Time per Frame: {average_elapsed_time:.5f} seconds")


if __name__ == "__main__":
    main()
    print("Covariance tracking completed.")

