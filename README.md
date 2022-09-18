__Updates:__ This repository will be used to release our code for the most recent ICRA 2023 submission, _DytanVO: Joint Refinement of Visual Odometry and Motion Segmentation in Dynamic Environments_. Please check out ```main``` for it.


# (ARCHIVED in this branch): Dynamic Dense RGB-D SLAM with Learning-Based Visual Odometry  
We propose a dense dynamic RGB-D SLAM pipeline based on a learning-based visual odometry, TartanVO. TartanVO, like other direct methods rather than feature-based, estimates camera pose through dense optical flow, which only applies to static scenes and disregards dynamic objects. Due to the color constancy assumption, optical flow is not able to differentiate between dynamic and static pixels. Therefore, to reconstruct a static map through such direct methods, our pipeline resolves dynamic/static segmentation by leveraging the optical flow output, and only fuse static points into the map. Moreover, we rerender the input frames such that the dynamic pixels are removed and iteratively pass them back into the visual odometry to refine the pose estimate. Please see our [paper](https://arxiv.org/abs/2205.05916) for more details.

## Datasets
Download [TUM fr3_walking_xyz](https://vision.in.tum.de/data/datasets/rgbd-dataset/download#freiburg3_walking_xyz) and place ```depth/```, ```rgb/```, ```depth.txt```, ```rgb.txt```, and ```groundtruth.txt``` under ```tartanvo/data/fr3_walking_xyz```. 

Run ```preprocess/associate.py``` to associate RGB and depth images as well as their corresponding groundtruth poses. This should create ```tartanvo/data/fr3_walking_xyz/associated```.

Run ```preprocess/crop.py``` to center crop RGB and depth images respectively so that the aspect ratio matches the optical flow dimension from PWC-Net. This should create ```depth/``` and ```rgb/``` under ```tartanvo/data/fr3_walking_xyz/cropped```.

*Optional*: Run ```preprocess/upscale.py``` to upscale the optical flow visualizations in ```tartanvo/results/fr3_tartanvo_1914_flow``` to match RGBD images' dimensions.

*Optional*: Run ```prerpocess/debug_data.py``` to test TartanVO over the first ```500``` frames which have the corresponding groundtruth segmentation mask inside ```tartanvo/data/fr3_walking_xyz/cropped/mask```. This can be used to compare the performance of TartanVO with and without closed-loop pose refinement using rerendered images. 

## Usage
Set up the frontend, TartanVO, by following [README](tartanvo/README.md).

Two methodologies of segmentation are ```segmentation1.py``` and ```segmentation2.py``` inside ```backend/```. The main includes the implementation of the entire backend, i.e., removing dynamic pixels from images and reconstructing only the static points through point-based fusion. 

```slam.py``` in ```tartanvo/``` implements the closed-loop pose refinement using rerendered images after iteratively removing dynamic pixels in backend.
