import os
import numpy as np
import shutil

# Train TartanVO over the first 500 frames
gt = np.loadtxt('../tartanvo/data/fr3_walking_xyz/associated/gt_pose.txt').astype(np.float32)
gt = gt[:501]
np.savetxt('../tartanvo/data/fr3_walking_xyz/associated/gt_pose100.txt', gt)

rgb_dir = '../tartanvo/data/fr3_walking_xyz/cropped/rgb'
imgs = os.listdir(rgb_dir)
imgs.sort()

for img in imgs[:501]:
    shutil.copy(os.path.join(rgb_dir, img), os.path.join('../tartanvo/data/fr3_walking_xyz/cropped/rgb_debug', img))