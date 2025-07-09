#!/bin/bash

# baseline
nohup taskset -c 0-10 python src/covariance_tracking.py --bounding_box_path dataset/groundtruth.txt > logs/covariance_tracking_baseline.log 2>&1 &

# feature extraction
#python src/covariance_tracking.py --bounding_box_path results/feature_extraction.txt
