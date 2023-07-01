conda activate dytanvo
tarj=00_1
python -W ignore::UserWarning vo_trajectory_from_folder.py --vo-model-name vonet_ft.pkl --seg-model-name segnet-kitti.pth --kitti --kitti-intrinsics-file data/DynaKITTI/$traj/calib.txt    --test-dir data/DynaKITTI/$traj/image_2  --pose-file data/DynaKITTI/$traj/pose_left.txt 


python vo_trajectory_from_folder.py --vo-model-name vonet_ft.pkl --seg-model-name segnet-kitti.pth --kitti --kitti-intrinsics-file data/DynaKITTI/$traj/calib.txt    --test-dir data/DynaKITTI/$traj/image_2  --pose-file data/DynaKITTI/$traj/pose_left.txt 
