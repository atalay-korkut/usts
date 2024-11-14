# USTS

Ultrasound Segmentation and Tracking Suite (USTS)

Code of my bachelor's thesis on "Semi-supervised Instance Segmentation and Tracking in Ultrasound Video Streams".

It uses 3 different tools for segmentation: SAM, SAMUS and SAMMed

### Setup

requirements.txt TODO

SAM Weights: https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints

SaMMed Weights: https://github.com/OpenGVLab/SAM-Med2D

SAMUS Weights: TODO

### Usage:

#### GUI:

SAMMed: Segment-and-Track-Anything/app_sammed_manual_box.py

SAM: Segment-and-Track-Anything/app_sammed_manual_box.py

SAMUS: Segment-and-Track-Anything/app.py

#### Terminal: 

SAMMed: Segment-and-Track-Anything/cli_sammedtrack.py

SAM: Segment-and-Track-Anything/cli_samtrack.py

SAMUS: Segment-and-Track-Anything/cli_samustrack.py

input_video: path of input video
point inputs x,y,mode (e.g., 100,100,1) ,The mode indicates whether the point is positive (1) or negative (0)
bbox inputs x1,y1,x2,y2 (e.g., 100,100;200,200)
