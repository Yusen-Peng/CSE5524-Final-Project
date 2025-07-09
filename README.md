# No initial Bounding Box is Needed: A Feature Extraction Driven Approach for Object Tracking 

## Motivation

Many existing object tracking methods require initital bounding boxes for object tracking. In our work, we propose an approach that automatically detects interest points using interest point detectors on the first frame image, performs segmentation techniques upon these detected interest points, and finally generate our own bounding boxes without relying on provided ones. Eventually, we perform multiple tracking techniques to track objects in the dataset. We provide detailed benchmark study and intuitive video demos as well.

## Environment Setup

We recommend settting up environment using Conda environment and our `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate cse_5524
```

This will install all required dependencies and activate the cse_5524 environment.

## Dataset

The dataset is a single video sequence with **597** frames downloaded from [Kaggle](https://www.kaggle.com/datasets/kmader/videoobjecttracking) (specifically, the `TrackingDataset/Other/woman` dataset). This dataset is designed for single object tracking, but it's still challenging because moving camera introduces changing and complicated background - which makes it a decent dataset for benchmark study!

## Interest Point Detection

TBD

## Interest Point Segmentation

TBD

## Object Tracking

We explore three basic object tracking algorithms:

- [x] covariance tracking
- [ ] mean shift tracking
- [ ] KLT tracking

### Efficiency Metric: Elapsed Time

We simply measure the elapsed time it takes to track a single frame on average, namely:

```math
\text{Average Elapsed Time per Frame} = \frac{T_{\text{total}}}{N}
```

Where **N** represents the total number of frames in the video sequence. 

### Performance Metric: Intersection over Union (IoU)

![alt text](docs/IoU.png)

Image credit: [PyImageSearch](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)

We are given a predicted bounding box and the ground-truth bounding box:

```math
\text{predicted bounding box}: (x_1^{\text{pred}},\ y_1^{\text{pred}},\ x_2^{\text{pred}},\ y_2^{\text{pred}})
```

```math
\text{ground-truth bounding box}: (x_1^{\text{gt}},\ y_1^{\text{gt}},\ x_2^{\text{gt}},\ y_2^{\text{gt}})
```

Intersection over Union (IoU) is a common evaluation metric for object tracking: it is computed as follows:

```math
\text{IoU} = \frac{\text{Area}_{\text{intersection}}}{\text{Area}_{\text{union}}}
```

In order to compute Area of Intersection, we will find the intersected box by computing the following 4 quantities first: 

```math
x_{\text{left}} = \max(x_1^{\text{gt}},\ x_1^{\text{pred}})
```

```math
y_{\text{top}} = \max(y_1^{\text{gt}},\ y_1^{\text{pred}})
```

```math
x_{\text{right}} = \min(x_2^{\text{gt}},\ x_2^{\text{pred}}) 
```

```math
y_{\text{bottom}} = \min(y_2^{\text{gt}},\ y_2^{\text{pred}})
```

Then we can calculate Area of Intersection:

```math
\text{Area}_{\text{intersection}} = (x_{\text{right}} - x_{\text{left}}) \cdot (\ y_{\text{bottom}} - y_{\text{top}})
```

we still need tocompute the area for each box:

```math
\text{Area}_{\text{gt}} = (x_2^{\text{gt}} - x_1^{\text{gt}}) \cdot (y_2^{\text{gt}} - y_1^{\text{gt}})
```

```math
\text{Area}_{\text{pred}} = (x_2^{\text{pred}} - x_1^{\text{pred}}) \cdot (y_2^{\text{pred}} - y_1^{\text{pred}})
```

Now, we can compute Area of Union:

```math
\text{Area}_{\text{union}} = \text{Area}_{\text{gt}} + \text{Area}_{\text{pred}} - \text{Area}_{\text{intersection}}
```

Eventually, we compute the average Intersection over Union (IoU) within a single frame on average:

```math
\text{Average Intersection over Union per Frame} = \frac{1}{N} \sum_{i=1}^{N} \text{IoU}_i
```

Where **N** represents the total number of frames in the video sequence.

### Benchmark Results

We compute average IoU for the first 20 frames, first 70 frames, and IoU for all 597 frames respectively: In the first 20 frames, the background barely changes; in the first 70 changes, the background does change but is still relatively easy; after the first 70 frames, more complex background gets introduced. We compute both IoU metrics to study how **sensitive** each tracking algorithm is to varying/challenging backgrounds. The average elapsed time is measured across all 597 frames, though.

| Tracking Method | IoU (first 20 frames) | IoU (first 70 frames) | IoU (all 597 frames) | Time (seconds) | 
| --------------- | --- | ---- | ---- | ---- |
| Covariance, baseline | **85.25%** | **44.44%** | **7.63%** | **26.9414** |
| Mean-shift, baseline | **11.19%** | **11.98%** | **1.56%** | **0.9094** |
| KLT, baseline | **TBD** | **TBD** | **TBD** | **TBD** |
| Covariance, ours | **TBD** | **TBD** | **TBD** | **TBD** |
| Mean-shift, ours | **TBD** | **TBD** | **TBD** | **TBD** |
| KLT, ours | **TBD** | **TBD** | **TBD** | **TBD** |

### Video Demo

In addition to benchmark study, we also compile a list of video demos to visualize our tracking results:

| method | video link |
| ------ | ---------- | 
| ground truth | [ground truth](figures/video_demo/ground_truth.mp4)|
| Covariance, baseline | [covariance baseline](figures/video_demo/covariance_baseline.mp4) |
| Mean-shift, baseline | [mean-shift baseline](figures/video_demo/meanshift_baseline.mp4) |
| KLT, baseline | **TBD** |
| Covariance, ours | **TBD** |
| Mean-shift, ours | **TBD** |
| KLT, ours | **TBD** |

## Contact

If you have any questions or suggestions, feel free to contact:

- Yuhan Duan (duan.418@osu.edu)
- Cen Gu (gu.1027@osu.edu)
- Yusen Peng (peng.1007@osu.edu)

Or describe it in Issues.
