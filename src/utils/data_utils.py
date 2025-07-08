from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_ground_truth(frame_index: int, image_dir: str, gt_path: str):
    """
        visualize the ground truth bounding box.
    """
    image_filename = f"{image_dir}/img{1000 + frame_index}.jpg"
    image = io.imread(image_filename)

    with open(gt_path, 'r') as f:
        lines = f.readlines()
    
    if frame_index - 1 >= len(lines):
        raise IndexError(f"Frame index {frame_index} out of bounds for {len(lines)} frames.")

    x1, y1, x2, y2 = map(int, lines[frame_index - 1].strip().split(','))

    # plot image and overlay rectangle
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image)
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f"Ground Truth Bounding Box - Frame {frame_index}")
    ax.axis('off')
    plt.savefig(f"figures/ground_truth/ground_truth_frame_{frame_index}.png")
    plt.close(fig)
