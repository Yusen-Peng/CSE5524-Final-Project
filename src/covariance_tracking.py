import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.tracking_utils import extract_features, riemannian_distance, save_prediction

def build_match_distance_matrix(image: np.ndarray,
    modelCovMatrix: np.ndarray,
    window_height: int,
    window_width: int
) -> np.ndarray:
    """
        Compute the match distance matrix for the given image and model covariance matrix.
    """
    
    H, W, _ = image.shape
    
    # intialize the match distance matrix
    match_distance_matrix = np.zeros(
        (H - window_height + 1, W - window_width + 1), 
        dtype=np.float32
    )

    for y in tqdm(range(H - window_height + 1)):
        for x in range(W - window_width + 1):
            
            # extract windows
            window = image[y:y + window_height, x:x + window_width]

            # feature extraction
            features = extract_features(window)

            # covariance matrix
            cov_mat = np.cov(features, rowvar=False, bias=True)

            # distance measure
            distance = riemannian_distance(modelCovMatrix, cov_mat)

            # save the match distance for each box location
            match_distance_matrix[y, x] = distance
    
    return match_distance_matrix


def visualize_match_distance_matrix(match_distance_matrix: np.ndarray, frame_index: int):

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(match_distance_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Riemannian Distance', orientation='horizontal')
    plt.title('Match Distance Matrix')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.savefig(f'figures/match_distance/matrix_{frame_index}.png')
    plt.close(fig)


def visualize_best_match(
        image: np.ndarray,
        frame_index: int,
        best_y: int,
        best_x: int,
        window_height: int,
        window_width: int
    ):
    """
        visualize the best match bounding box on the original image.
    """

    # for visualization, draw bounding box on the original image
    image_with_box = image.copy().astype(np.uint8)
    rr, cc = best_y, best_x
    GREEN = [0, 255, 0]
    image_with_box[rr:rr+window_height, cc, :] = GREEN
    image_with_box[rr:rr+window_height, cc+window_width-1, :] =  GREEN
    image_with_box[rr, cc:cc+window_width, :] = GREEN
    image_with_box[rr+window_height-1, cc:cc+window_width, :] = GREEN

    # save result image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_with_box.astype(np.uint8))
    plt.title("Best Match Bounding Box")
    plt.axis('off')
    plt.savefig(f'figures/covariance_tracking/best_match_frame_{frame_index}.png')
    plt.close()


def covariance_tracking(
        image: np.ndarray,
        frame_index: int,
        modelCovMatrix: np.ndarray, 
        window_height: int, 
        window_width: int,
        verbose: bool = True
    ):
    """
        Perform covariance tracking on the given image.
    """

    # compute the match distance matrix
    match_distance_matrix = build_match_distance_matrix(image=image,
        modelCovMatrix=modelCovMatrix,
        window_height=window_height,
        window_width=window_width
    )

    if verbose:
        # visualize the match distance matrix (optional)
        print("Visualizing match distance matrix...")
        visualize_match_distance_matrix(match_distance_matrix, frame_index)


    # provide the location of the best candidate
    best_y, best_x = np.unravel_index(
        np.argmin(match_distance_matrix), 
        match_distance_matrix.shape
    )


    if verbose:
        # visualize the best match (optional)
        print("Visualizing best match...")
        visualize_best_match(image=image,
                            best_y=best_y,
                            best_x=best_x,
                            window_height=window_height,
                            window_width=window_width)
    
    return best_y, best_x


