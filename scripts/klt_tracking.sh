#!/bin/bash

# baseline
nohup taskset -c 21-30 python src/KLT_tracking.py --bounding_box_path dataset/groundtruth.txt > logs/KLT_tracking_baseline.log 2>&1 &

# feature extraction
#python src/KLT_tracking.py --bounding_box_path results/feature_extraction.txt
