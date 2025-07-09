#!/bin/bash

# baseline
nohup taskset -c 11-20 python src/mean_shift_tracking.py --bounding_box_path dataset/groundtruth.txt > logs/mean_shift_tracking_baseline.log 2>&1 &

# feature extraction
#python src/mean_shift_tracking.py --bounding_box_path results/feature_extraction.txt
