import cv2
import os
import glob
from natsort import natsorted


def demo_maker(image_folder: str, output_path: str, fps: int, descriptor: str):
    """
        Create a video from a folder of images.
    """

    if descriptor != 'ground_truth':
        image_paths = natsorted(glob.glob(os.path.join(image_folder, '*.jpg')))
    else:
        image_paths = natsorted(glob.glob(os.path.join(image_folder, '*.png')))

    # read first image to get frame size
    first_frame = cv2.imread(image_paths[0])
    height, width, _ = first_frame.shape
    frame_size = (width, height)

    # prepare the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = f"{output_path}/{descriptor}.mp4"
    out = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
    for _, img_path in enumerate(image_paths):
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    print(f"Video saved to {save_path}")


if __name__ == "__main__":
    #image_folder = 'dataset/frames_with_boxes'
    #image_folder = 'figures/covariance_tracking/'
    image_folder = 'figures/meanshift_tracking'
    output_path = 'figures/video_demo'
    fps = 30
    descriptor = 'meanshift_baseline' # 'covariance_baseline', 'meanshift_baseline' or 'ground_truth'
    demo_maker(image_folder, output_path, fps, descriptor)
