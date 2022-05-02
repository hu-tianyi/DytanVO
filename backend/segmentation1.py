'''
Segmentation Methodology 1: 2D Scene Flow Dynamic Segmentation
'''


import sys
sys.path.append('./point_fusion')

import numpy as np
from utils import visflow, dataset_intrinsics
import glob
import cv2
import transformation as tf
import open3d as o3d
# from transforms import unproject, project
import matplotlib.pyplot as plt
import matplotlib
from helper import *
import seaborn as sns
from point_fusion.fusion import Map
from point_fusion.transforms import *
import point_fusion.o3d_utility as o3d_utility
matplotlib.use('TKAgg')


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
        # T2 = np.eye(4)
        # T2[:3, :3] = T1[:3, :3] @ T0[:3, :3].T
        # T2[:3, -1] = T1[:3, -1] - T2[:3, :3] @ T0[:3, -1]
        T =  T1 @ np.linalg.pinv(T0)
        res.append(T)  # T is the same as T1 @ inv(T0)
    return res


def warp(depth_map0, K, T):
    h, w = depth_map0.shape
    _us, _vs = np.meshgrid(np.arange(w)-w/2, np.arange(h)-h/2)
    points = np.stack((_us, _vs, np.ones((h, w))), axis=2)
    points = points.reshape(-1, 3)

    K = np.hstack((K, np.zeros((3,1))))
    points_warped =  K @ T @ np.linalg.pinv(K) @ (points.T)  # 3 x (h*w)
    # points_warped =  T @ (points.T)  # 3 x (h*w)
    # points_warped = K @ (T[:3, :3].T @ (np.linalg.inv(K) @ points.T) -  T[:3, -1].reshape(-1, 1)) # 3 x (h*w)

    us = (points_warped[0, :] / points_warped[2, :]).reshape(h, w)
    vs = (points_warped[1, :] / points_warped[2, :]).reshape(h, w)
    
    ego_flow = np.stack((us - _us, vs - _vs), axis=2)  # h X w X 2
    return ego_flow

def warp_usedepth(depth_map0, depth_map1, K, T0, T1):
    h, w = depth_map0.shape
    _us, _vs = np.meshgrid(np.arange(w), np.arange(h))

    pc0 = unproject(depth_map0,K).reshape(-1,3)
    pc0 = np.hstack((pc0, np.ones((h*w, 1))))
    T = T0 @ np.linalg.inv(T1)
    # T =np.eye(4)
    pc1 = (T @ pc0.T).T
    us, vs, ds = project(pc1, K)

    us = np.round(us).astype(int)
    vs = np.round(vs).astype(int)
    filter_mask = (us >= 0) & (us < w) & (vs >= 0) & (vs < h) & (ds >= 0)

    indices = np.arange(len(pc0)).astype(int)
    indices = indices[filter_mask]
    us[~filter_mask] = _us.reshape(h*w)[~filter_mask]
    vs[~filter_mask] = _vs.reshape(h*w)[~filter_mask]
    # us = us[filter_mask]
    # vs = vs[filter_mask]
    
    us = (us).reshape(h, w)
    vs = (vs).reshape(h, w)
    filter_mask = filter_mask.reshape(h,w)

    ego_flow = np.stack((us - _us, vs - _vs), axis=2)  # h X w X 2    return ego_flow
    return ego_flow, filter_mask

def np_depth2flow(depth, K_ini, T0, T1):
    """ Numpy implementation.
    Estimate the ego-motion flow given two frames depths and transformation matrices. 
    The output is an ego-motion flow in 2D (2*H*W).
    :param the depth map of the reference frame
    :param the camera intrinsics
    :param the camera coordinate of the reference frame
    :param the camera coordinate of the target frame
    """
    h, w = depth.shape
    u_mat = np.repeat(np.array(range(0, w)).reshape(1, w), h, axis=0)
    v_mat = np.repeat(np.array(range(0, h)).reshape(h, 1), w, axis=1)

    K = np.eye(4)
    K[0,0], K[1,1], K[0,2], K[1,2] = K_ini[0,0], K_ini[1,1], K_ini[0,2], K_ini[1,2]
    # inv_k = [ 1/f_x,  0, -c_x/f_x, 0;
    #           0,  1/f_y, -c_y/f_y, 0;
    #           0,      0,  1,       0;
    #           0,      0,  0,       1]
    inv_K = np.eye(4)
    inv_K[0,0], inv_K[1,1] = 1.0 / K[0,0], 1.0 / K[1,1]
    inv_K[0,2], inv_K[1,2] = -K[0,2] / K[0,0], -K[1,2] / K[1,1]

    # the point cloud move w.r.t. the inverse of camera transform
    T = K @ T1 @ np.linalg.inv(T0) @ inv_K

    # blender's coordinate is different from sintel
    ones = np.ones((h, w))
    z = depth
    x = depth * u_mat
    y = depth * v_mat
    p4d = np.dstack((x, y, z, ones)).transpose((2,0,1))
    p4d_t = np.tensordot(T, p4d, axes=1)

    x_t, y_t, z_t, w_t = np.split(p4d_t, 4)
    # homogenous to cartsian
    x_t, y_t, z_t = x_t[0] / w_t[0], y_t[0] / w_t[0], z_t[0] / w_t[0]

    u_t_mat = x_t / z_t
    v_t_mat = y_t / z_t

    # this is the final ego-motion flow
    d_u = u_t_mat - u_mat
    d_v = v_t_mat - v_mat

    # d_u[d_u>w] =0
    # d_u[d_u<-w] =0
    # d_v[d_v>h] =0
    # d_v[d_v<-h] =0
    ego_flow = np.stack((d_u, d_v), axis=2)
    return  ego_flow, u_t_mat, v_t_mat, z_t



if __name__ == "__main__":
    cropped_dir = "../tartanvo/data/fr3_walking_xyz/cropped/"
    flow_dir = "../tartanvo/results/fr3_tartanvo_1914_flow/"
    # upscale_flow(flow_dir, cropped_dir)
    
    fx, fy, cx, cy = dataset_intrinsics("fr3")
    intrinsics = np.eye(3)
    intrinsics[0, 0] = fx
    intrinsics[1, 1] = fy
    intrinsics[0, 2] = cx
    intrinsics[1, 2] = cy

    pose_file = "data/fr3_walking_xyz/associated/gt_pose_2.txt"
    poses_in_world = np.loadtxt(pose_file)
    poses_camera_SEs = get_camera_poses(poses_in_world)
    SEs = tf.pos_quats2SE_matrices(poses_in_world)

    m = Map()

    depth_scale = 5000  # TUM convention for depth images
    for i in range(len(glob.glob1(cropped_dir + "depth", "*.png")) - 1 ):
        depth0 = o3d.io.read_image(cropped_dir + "depth/" + str(i).zfill(4) + ".png")
        depth1 = o3d.io.read_image(cropped_dir + "depth/" + str(i+1).zfill(4) + ".png")

        depth0 = np.asarray(depth0) / depth_scale
        depth1 = np.asarray(depth1) / depth_scale
        h, w = depth0.shape

        # ego_flow = warp(depth0, intrinsics,poses_camera_SEs[i])
        ego_flow, filter_mask = warp_usedepth(depth0, depth1, intrinsics,SEs[i], SEs[i+1])
        # ego_flow,_,_,_= np_depth2flow(depth0, intrinsics, SEs[i], SEs[i+1])

        opt_flow = np.load(flow_dir + str(i).zfill(6) + ".npy" )
        opt_flow = cv2.resize(opt_flow, (640, 448), interpolation=cv2.INTER_LINEAR)
        # print(np.max(opt_flow))
        # print(np.mean(opt_flow))
        # print(np.max(ego_flow))
        # print(np.mean(ego_flow))
        sce_flow = opt_flow - ego_flow

        # f, ax = plt.subplots(2, 2)
        # ax[0,0].imshow(o3d.io.read_image(cropped_dir + "rgb/" + str(i).zfill(4) + ".png"))
        # ax[0,1].imshow(o3d.io.read_image(cropped_dir + "rgb/" + str(i+1).zfill(4) + ".png"))
        # ax[1,0].imshow(visflow(opt_flow))
        # ax[1,1].imshow(visflow(sce_flow))
        # plt.show()

        # rgb = cv2.imread(cropped_dir + "rgb/" + str(i).zfill(4) + ".png")
        # for i in range(0, len(depth0), 16):
        #     for j in range(0, len(depth0[0]), 16):
        #         u, v = 4 * opt_flow[i, j, :]
        #         rgb = cv2.arrowedLine(rgb, (int(j), int(i)), (int(j + v), int(i + u)), (0, 255, 255), 1, line_type = cv2.LINE_AA, tipLength=0.2)
        #         u, v = 4 * ego_flow[i, j, :]
        #         rgb = cv2.arrowedLine(rgb, (int(j), int(i)), (int(j + v), int(i + u)), (255, 0, 0), 1, line_type = cv2.LINE_AA, tipLength=0.2)
        # cv2.imshow('img',rgb)

        # break

        # mask for dynamic object
        sce_flow = sce_flow.reshape(-1,2)
        dis_sce_flow = np.linalg.norm(sce_flow, axis=1)
        print(np.max(dis_sce_flow))
        print(np.mean(dis_sce_flow))
        static_mask = dis_sce_flow < np.mean(dis_sce_flow) *1.3
        # mask = filter_mask.reshape(-1) & static_mask
        rgb = cv2.imread(cropped_dir + "rgb/" + str(i).zfill(4) + ".png")
        rgb = cv2.bitwise_and(rgb, rgb, mask=static_mask.reshape(h, w).astype(np.uint8))
        cv2.imshow("masked", rgb)
        cv2.waitKey(1)
        
        # dis_sce_flow_map = dis_sce_flow.reshape(h, w)
        # ax = sns.heatmap(dis_sce_flow_map)
        # plt.show(block=False)

        # fusion
        source_vertex_map = unproject(depth0, intrinsics)
        source_color_map = np.asarray(o3d.io.read_image(cropped_dir + "rgb/" + str(i).zfill(4) + ".png")).astype(float) / 255.0
        valid_depth_mask = source_vertex_map[:,:,2] > 0.001

        pcd = o3d_utility.make_point_cloud(source_vertex_map[valid_depth_mask].reshape((-1, 3)))
        pcd.estimate_normals()

        source_normal_map = np.zeros_like(source_vertex_map)
        source_normal_map[valid_depth_mask] = np.asarray(pcd.normals)
        m.fuse_bonus(source_vertex_map, source_normal_map, source_color_map,
                    intrinsics, SEs[i+1], i,t_max=5, static_mask=static_mask, 
                    depth_mask=valid_depth_mask.flatten())

        # break
    global_pcd = o3d_utility.make_point_cloud(m.points,
                                              colors=m.colors,
                                              normals=m.normals)
    o3d.visualization.draw_geometries(
        [global_pcd.transform(o3d_utility.flip_transform)])