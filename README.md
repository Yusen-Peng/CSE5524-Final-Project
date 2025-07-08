# No initial Bounding Box is Needed: A Feature Extraction Driven Approach for Object Tracking 

## CSE5524 Final Project - Yuhan Duan, Cen Gu, Yusen Peng

Many existing object tracking methods require initital bounding boxes for object tracking. In our work, we propose an approach that automatically detects interest points using interest point detectors on the first frame image, performs segmentation techniques upon these detected interest points, and finally generate our own bounding boxes without relying on provided ones. Eventually, we perform multiple tracking techniques to track objects in the dataset.

## Environment Setup

TBD

## Dataset Preparation

TBD

## Interest Point Detection

TBD

## Interest Point Segmentation

TBD

## Object Tracking

TBD

## Video Demo

TBD

## Benchmark Study

### Metric: Intersection over Union (IoU)

We are given a predicted bounding box and the ground-truth bounding box:

```math
\text{predicted bounding box}: (x_1^{\text{pred}},\ y_1^{\text{pred}},\ x_2^{\text{pred}},\ y_2^{\text{pred}}) \\
\text{ground-truth bounding box}: (x_1^{\text{gt}},\ y_1^{\text{gt}},\ x_2^{\text{gt}},\ y_2^{\text{gt}})
```

Intersection over Union (IoU) is a common evaluation metric for object tracking: it is computed as follows:

```math
\text{IoU} = \frac{\text{Area}_{\text{union}}}{\text{Area}_{\text{intersection}}}
```

In order to compute Area of Intersection, we will find the intersected box by computing the following 4 quantities first: 

```math
x_{\text{left}} = \max(x_1^{\text{gt}},\ x_1^{\text{pred}}) \\
y_{\text{top}} = \max(y_1^{\text{gt}},\ y_1^{\text{pred}}) \\
x_{\text{right}} = \min(x_2^{\text{gt}},\ x_2^{\text{pred}}) \\ 
y_{\text{bottom}} = \min(y_2^{\text{gt}},\ y_2^{\text{pred}}) \\
```

Then we can calculate Area of Intersection:

```math
\text{Area}_{\text{intersection}} = (x_{\text{right}} - x_{\text{left}}) \cdot (\ y_{\text{bottom}} - y_{\text{top}})
```

we still need tocompute the area for each box:

```math
\text{Area}_{\text{gt}} = (x_2^{\text{gt}} - x_1^{\text{gt}}) \cdot (y_2^{\text{gt}} - y_1^{\text{gt}}) \\
\text{Area}_{\text{pred}} = (x_2^{\text{pred}} - x_1^{\text{pred}}) \cdot (y_2^{\text{pred}} - y_1^{\text{pred}})
```

Eventually, we can compute Area of Union:

```math
\text{Area}_{\text{union}} = \text{Area}_{\text{gt}} + \text{Area}_{\text{pred}} - \text{Area}_{\text{intersection}}
```
