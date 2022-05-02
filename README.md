# Dynamic-Dense-RGBD-SLAM-with-TartanVO

## Datasets
Download [TUM fr3_walking_xyz](https://vision.in.tum.de/data/datasets/rgbd-dataset/download#freiburg3_walking_xyz) and place ```depth/```, ```rgb/```, ```depth.txt```, ```rgb.txt```, and ```groundtruth.txt``` under ```tartanvo/data/fr3_walking_xyz```. 

Run ```preprocess/associate.py``` to associate RGB and depth images as well as their corresponding groundtruth poses. This should create ```tartanvo/data/fr3_walking_xyz/associated```.

Run ```preprocess/crop.py``` to center crop RGB and depth images respectively so that the aspect ratio matches the optical flow dimension from PWC-Net. This should create ```depth/``` and ```rgb/``` under ```tartanvo/data/fr3_walking_xyz/cropped```.

*Optional*: Run ```preprocess/upscale.py``` to upscale the optical flow visualizations in ```tartanvo/results/fr3_tartanvo_1914_flow``` to match RGBD images' dimensions.

*Optional*: Run ```prerpocess/debug_data.py``` to test TartanVO over the first ```500``` frames which have the corresponding groundtruth segmentation mask inside ```tartanvo/data/fr3_walking_xyz/cropped/mask```. This can be used to compare the performance of TartanVO with and without closed-loop pose refinement using rerendered images. 

## Usage
Two methodologies of segmentation are ```segmentation1.py``` and ```segmentation2.py``` inside ```backend/```. The main includes the implementation of the entire backend, i.e., removing dynamic pixels from images and reconstructing only the static points through point-based fusion. 

```slam.py``` in ```tartanvo/``` implements the closed-loop pose refinement using rerendered images after iteratively removing dynamic pixels in backend.
