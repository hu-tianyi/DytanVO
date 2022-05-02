'''
Segmentation Methodology 2: Epipolar Line Based Motion Consistency Detection
'''


import sys
sys.path.append('./point_fusion')

import numpy as np
from utils import visflow, dataset_intrinsics, calculate_angle_distance_from_du_dv
import glob
import cv2
import transformation as tf
import open3d as o3d
import matplotlib.pyplot as plt
from helper import *
import seaborn as sns
from point_fusion.fusion import Map
from point_fusion.transforms import *
import point_fusion.o3d_utility as o3d_utility

def upscale_flow(flow_dir, save_dir):
    num_flow = len(glob.glob1(flow_dir, "*.npy"))
    
    for i in range(num_flow):
        flow = np.load(flow_dir + str(i).zfill(6) + ".npy")
        flow_upscaled = cv2.resize(flow, (640, 448), interpolation=cv2.INTER_LINEAR)
        flow_vis = visflow(flow_upscaled)
        cv2.imwrite(save_dir + "flow/" + str(i).zfill(6) + ".png", flow_vis)
    

def get_camera_poses(poses_in_world):
    SEs = tf.pos_quats2SE_matrices(poses_in_world)
    res = []
    for i in range(len(SEs) - 1):
        T0, T1 = SEs[i], SEs[i+1]
        # T = np.eye(4)
        # T[:3, :3] = T1[:3, :3] @ T0[:3, :3].T
        # T[:3, -1] = T1[:3, -1] - T[:3, :3] @ T0[:3, -1]
        T = T1 @ np.linalg.inv(T0)
        res.append(T)  # T is the same as T1 @ inv(T0)
    return res


def warp(depth_map, K, T):
    h, w = depth_map.shape
    _us, _vs = np.meshgrid(np.arange(w), np.arange(h))
    points = np.stack((_us, _vs, np.ones((h, w))), axis=2)
    points = points.reshape(-1, 3)
    points_warped = K @ (T[:3, :3] @ (np.linalg.inv(K) @ points.T) + T[:3, -1].reshape(-1, 1)) # 3 x (h*w)
    us = (points_warped[0, :] / points_warped[2, :]).reshape(h, w)
    vs = (points_warped[1, :] / points_warped[2, :]).reshape(h, w)
    
    ego_flow = np.stack((vs - _vs, us - _us), axis=2)  # h X w X 2
    return ego_flow


if __name__ == "__main__":
    cropped_dir = "../tartanvo/data/fr3_walking_xyz/cropped/"
    flow_dir = "../tartanvo/results/fr3_tartanvo_1914_flow/"
    # upscale_flow(flow_dir, cropped_dir)
    
    pose_file = "fr3_walking_xyz/associated/gt_pose.txt"
    poses_in_world = np.loadtxt(pose_file)
    poses_in_world_SEs = tf.pos_quats2SE_matrices(poses_in_world)

    poses_camera_SEs = get_camera_poses(poses_in_world)

    fx, fy, cx, cy = dataset_intrinsics("fr3")
    intrinsics = np.eye(3)
    intrinsics[0, 0] = fx
    intrinsics[1, 1] = fy
    intrinsics[0, 2] = cx
    intrinsics[1, 2] = cy
    
    depth_scale = 5000  # TUM convention for depth images
    dist_thresh = 0.0025
    m = Map()
    # for i in range(len(glob.glob1(cropped_dir + "depth", "*.png")) - 1):
    for i in range(485):
        print("Fusing frame %d..." % i)
        depth = o3d.io.read_image(cropped_dir + "depth/" + str(i).zfill(4) + ".png")
        depth = np.asarray(depth) / depth_scale
        h, w = depth.shape

        # ego_flow = warp(depth, intrinsics, poses_camera_SEs[i])
        opt_flow = np.load(flow_dir + str(i).zfill(6) + ".npy")
        opt_flow = cv2.resize(opt_flow, (640, 448), interpolation=cv2.INTER_LINEAR)
        # du, dv = opt_flow[:,:,0], opt_flow[:,:,1]
        # mag = np.sqrt(du * du + dv * dv)
        # opt_flow = opt_flow / np.max(mag)

        # print(np.max(opt_flow))
        # print(np.mean(opt_flow))
        # print(np.max(ego_flow))
        # print(np.mean(ego_flow))
        # sce_flow = opt_flow - ego_flow

        # f, ax = plt.subplots(2, 2)
        # ax[0,0].imshow(o3d.io.read_image(cropped_dir + "rgb/" + str(i).zfill(4) + ".png"))
        # ax[0,1].imshow(o3d.io.read_image(cropped_dir + "rgb/" + str(i+1).zfill(4) + ".png"))
        # ax[1,0].imshow(visflow(opt_flow))
        # ax[1,1].imshow(visflow(sce_flow))
        # plt.show()

        # rgb = cv2.imread(cropped_dir + "rgb/" + str(i).zfill(4) + ".png")
        # for i in range(0, len(depth), 16):
        #     for j in range(0, len(depth[0]), 16):
        #         u, v = 4 * opt_flow[i, j, :]
        #         rgb = cv2.arrowedLine(rgb, (int(j), int(i)), (int(j + v), int(i + u)), (0, 255, 255), 1, line_type = cv2.LINE_AA, tipLength=0.2)
        #         u, v = 4 * ego_flow[i, j, :]
        #         rgb = cv2.arrowedLine(rgb, (int(j), int(i)), (int(j + v), int(i + u)), (255, 0, 0), 1, line_type = cv2.LINE_AA, tipLength=0.2)
                
        # plt.imshow(rgb)
        # plt.show()

        flow = cv2.imread(cropped_dir + "flow/" + str(i).zfill(6) + ".png")
        cv2.imshow("Flow", flow)
        cv2.waitKey(1)

        ang, mag, _ = calculate_angle_distance_from_du_dv(opt_flow[:,:,0], opt_flow[:,:,1])
        if np.var(ang) < 0.0:
            mask = None
            rgb = cv2.imread(cropped_dir + "rgb/" + str(i).zfill(4) + ".png")
            cv2.imshow("Segmented", rgb)
            cv2.waitKey(1)
        else:
            T = poses_camera_SEs[i]
            R, t = T[:3, :3], T[:3, -1]
            F = np.linalg.inv(intrinsics.T) @ (skew(t) @ R) @ np.linalg.inv(intrinsics)
            # im0 = cv2.imread(cropped_dir + "rgb/" + str(i).zfill(4) + ".png")
            # im1 = cv2.imread(cropped_dir + "rgb/" + str(i+1).zfill(4) + ".png")
            # epipolarMatchGUI(im0[:,:,::-1], im1[:,:,::-1], F)

            u0s, v0s = np.meshgrid(np.arange(w), np.arange(h))
            p0s = np.stack((u0s, v0s, np.ones((h, w))), axis=2).reshape(-1, 3)
            l0s = (F @ p0s.T).T

            u1s, v1s = u0s + opt_flow[:,:,0], v0s + opt_flow[:,:,1]
            p1s = np.stack((u1s, v1s, np.ones((h, w))), axis=2).reshape(-1, 3)

            ds = np.abs(np.sum(p1s * l0s, axis=1))
            ds = np.divide(ds, np.linalg.norm(l0s[:, :2], axis=1))
            ds = ds / np.linalg.norm(ds)
            ds_map = ds.reshape(h, w)
            # ax = sns.heatmap(ds_map)
            # plt.show(block=False)
            mask = ds < dist_thresh  # geometric distance from point to epipolar line

            rgb = cv2.imread(cropped_dir + "rgb/" + str(i).zfill(4) + ".png")
            rgb = cv2.bitwise_and(rgb, rgb, mask=mask.reshape(h, w).astype(np.uint8))
            cv2.imshow("Segmented", rgb)
            cv2.waitKey(1)
        
        source_vertex_map = unproject(depth, intrinsics)
        source_color_map = np.asarray(o3d.io.read_image(cropped_dir + "rgb/" + str(i).zfill(4) + ".png")).astype(float) / 255.0
        valid_depth_mask = source_vertex_map[:,:,2] > 0.01

        pcd = o3d_utility.make_point_cloud(source_vertex_map[valid_depth_mask].reshape((-1, 3)))
        pcd.estimate_normals()

        source_normal_map = np.zeros_like(source_vertex_map)
        source_normal_map[valid_depth_mask] = np.asarray(pcd.normals)
        pcd_img = m.fuse_bonus(source_vertex_map, source_normal_map, source_color_map,
                    intrinsics, poses_in_world_SEs[i], i, static_mask=mask, 
                    depth_mask=valid_depth_mask.flatten())
        cv2.imshow("Rendered", pcd_img)
        cv2.waitKey(0)

        break


    # global_pcd = o3d_utility.make_point_cloud(m.points,
    #                                           colors=m.colors,
    #                                           normals=m.normals)
    # o3d.visualization.draw_geometries(
    #     [global_pcd.transform(o3d_utility.flip_transform)])