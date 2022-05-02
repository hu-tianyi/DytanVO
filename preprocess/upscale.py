import sys
sys.path.append('../backend')

import numpy as np
import glob
import cv2
from utils import visflow


def upscale_flow(flow_dir, save_dir):
    num_flow = len(glob.glob1(flow_dir, "*.npy"))
    
    for i in range(num_flow):
        flow = np.load(flow_dir + str(i).zfill(6) + ".npy")
        flow_upscaled = cv2.resize(flow, (640, 448), interpolation=cv2.INTER_LINEAR)
        flow_vis = visflow(flow_upscaled)
        cv2.imwrite(save_dir + "flow/" + str(i).zfill(6) + ".png", flow_vis)


if __name__ == "__main__":
    cropped_dir = "../tartanvo/data/fr3_walking_xyz/cropped/"
    flow_dir = "../tartanvo/results/fr3_tartanvo_1914_flow/"
    # upscale_flow(flow_dir, cropped_dir)