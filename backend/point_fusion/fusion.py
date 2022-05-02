'''
    Initially written by Ming Hsiao in MATLAB
    Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
    Implemented by Shihao Shen
'''
import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import quaternion  # pip install numpy-quaternion

import transforms
import o3d_utility
import scipy.misc
from PIL import Image

from preprocess import load_gt_poses


class Map:
    def __init__(self):
        self.points = np.empty((0, 3))
        self.normals = np.empty((0, 3))
        self.colors = np.empty((0, 3))
        self.weights = np.empty((0, 1))
        self.timestamps = np.empty((0, 1))  # last timestamp the point is active (merged, or added)
        self.initialized = False

    def merge(self, indices, points, normals, colors, R, t, timestamp=-1):
        '''
        Implement the merge function
        \param self The current maintained map
        \param indices Indices of selected points. Used for IN PLACE modification.
        \param points Input associated points, (N, 3)
        \param normals Input associated normals, (N, 3)
        \param colors Input associated colors, (N, 3)
        \param R rotation from camera (input) to world (map), (3, 3)
        \param t translation from camera (input) to world (map), (3, )
        \param timestamp the current timestamp
        \return None, update map properties IN PLACE
        '''
        # transform from camera to world
        points_w = (R @ points.T + t).T
        normals_w = (R @ normals.T).T

        # weighted average
        weights = self.weights[indices]
        self.points[indices, :] = (self.points[indices, :] * weights + points_w) / (weights + 1)
        temp_normals = (self.normals[indices, :] * weights + normals_w) / (weights + 1)
        self.normals[indices, :] = temp_normals / np.linalg.norm(temp_normals, axis=1, keepdims=True)
        self.colors[indices, :] = (self.colors[indices, :] * weights + colors) / (weights + 1)

        # update weights, which is simply a counter of matched points
        self.weights[indices] = self.weights[indices] + 1

        # update timestamps, to distinguish between dormant and active points
        self.timestamps[indices] = timestamp


    def add(self, points, normals, colors, R, t, timestamp=-1):
        '''
        Implement the add function
        \param self The current maintained map
        \param points Input associated points, (N, 3)
        \param normals Input associated normals, (N, 3)
        \param colors Input associated colors, (N, 3)
        \param R rotation from camera (input) to world (map), (3, 3)
        \param t translation from camera (input) to world (map), (3, )
        \param timestamp the current timestamp
        \return None, update map properties by concatenation
        '''
        self.points = np.concatenate((self.points, (R @ points.T + t).T))
        self.normals = np.concatenate((self.normals, (R @ normals.T).T))
        self.colors = np.concatenate((self.colors, colors))
        self.weights = np.concatenate((self.weights, np.ones((len(points), 1))))
        self.timestamps = np.concatenate((self.timestamps, np.ones((len(points), 1)) * timestamp))


    def remove(self, c_stable, t_max, timestamp):
        '''
        Remove points that remain in the unstable state for a long time
        \param c_stable threshold to determine whether points are stable
        \param t_max threshold to determine whether points are dormant
        \param timestamp the current timestamp
        \return None, update map properties by removing outliers or artifacts
        '''
        # points with confidence counter (weights) >= c_stable are considered stable
        # points that haven't been active longer than t_max are considered dormant
        mask = (self.weights < c_stable) & ((timestamp - self.timestamps) > t_max)
        indices = np.arange(len(self.points)).astype(int)
        indices = indices[~(mask.squeeze())]
        self.points = self.points[indices]
        self.normals = self.normals[indices]
        self.colors = self.colors[indices]
        self.weights = self.weights[indices]
        self.timestamps = self.timestamps[indices]


    def filter_pass1(self, us, vs, ds, h, w):
        '''
        Implement the filter function
        \param self The current maintained map, unused
        \param us Putative corresponding u coordinates on an image, (N, 1)
        \param vs Putative corresponding v coordinates on an image, (N, 1)
        \param ds Putative corresponding d depth on an image, (N, 1)
        \param h Height of the image projected to
        \param w Width of the image projected to
        \return mask (N, 1) in bool indicating the valid coordinates
        '''
        
        return (us >= 0) & (us < w) & (vs >= 0) & (vs < h) & (ds >= 0)


    def filter_pass2(self, points, normals, input_points, input_normals,
                     dist_diff, angle_diff):
        '''
        Implement the filter function
        \param self The current maintained map, unused
        \param points Maintained associated points, (M, 3)
        \param normals Maintained associated normals, (M, 3)
        \param input_points Input associated points, (M, 3)
        \param input_normals Input associated normals, (M, 3)
        \param dist_diff Distance difference threshold to filter correspondences by positions
        \param angle_diff Angle difference threshold to filter correspondences by normals
        \return mask (N, 1) in bool indicating the valid correspondences
        '''
        d = (input_points - points) *  input_normals
        mask1 = np.linalg.norm(d, axis=1) < dist_diff
        
        n1 = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        n2 = input_normals / np.linalg.norm(input_normals, axis=1, keepdims=True)
        mask2 = np.arccos(np.sum(n1 * n2, axis=1)) < angle_diff
        
        return mask1 & mask2


    def fuse(self,
             vertex_map,
             normal_map,
             color_map,
             intrinsic,
             T,
             dist_diff=0.03,
             angle_diff=np.deg2rad(5)):
        '''
        \param self The current maintained map
        \param vertex_map Input vertex map, (H, W, 3)
        \param normal_map Input normal map, (H, W, 3)
        \param intrinsic Intrinsic matrix, (3, 3)
        \param T transformation from camera (input) to world (map), (4, 4)
        \return None, update map properties on demand
        '''
        # Camera to world
        R = T[:3, :3]
        t = T[:3, 3:]

        # World to camera
        T_inv = np.linalg.inv(T)
        R_inv = T_inv[:3, :3]
        t_inv = T_inv[:3, 3:]

        if not self.initialized:
            points = vertex_map.reshape((-1, 3))
            normals = normal_map.reshape((-1, 3))
            colors = color_map.reshape((-1, 3))

            # TODO: add step
            self.add(points, normals, colors, R, t)
            self.initialized = True

        else:
            h, w, _ = vertex_map.shape

            # Transform from world to camera for projective association
            indices = np.arange(len(self.points)).astype(int)
            T_points = (R_inv @ self.points.T + t_inv).T
            R_normals = (R_inv @ self.normals.T).T

            # Projective association
            us, vs, ds = transforms.project(T_points, intrinsic)
            us = np.round(us).astype(int)
            vs = np.round(vs).astype(int)

            # TODO: first filter: valid projection
            mask = self.filter_pass1(us, vs, ds, h, w)
            # Should not happen -- placeholder before implementation
            if mask.sum() == 0:
                return
            # End of TODO

            indices = indices[mask]
            us = us[mask]
            vs = vs[mask]

            T_points = T_points[indices]
            R_normals = R_normals[indices]
            valid_points = vertex_map[vs, us]
            valid_normals = normal_map[vs, us]

            # TODO: second filter: apply thresholds
            mask = self.filter_pass2(T_points, R_normals, valid_points,
                                     valid_normals, dist_diff, angle_diff)
            # Should not happen -- placeholder before implementation
            if mask.sum() == 0:
                return
            # End of TODO

            indices = indices[mask]
            us = us[mask]
            vs = vs[mask]

            updated_entries = len(indices)

            merged_points = vertex_map[vs, us]
            merged_normals = normal_map[vs, us]
            merged_colors = color_map[vs, us]

            # TODO: Merge step - compute weight average after transformation
            self.merge(indices, merged_points, merged_normals, merged_colors,
                       R, t)
            # End of TODO

            associated_mask = np.zeros((h, w)).astype(bool)
            associated_mask[vs, us] = True
            new_points = vertex_map[~associated_mask]
            new_normals = normal_map[~associated_mask]
            new_colors = color_map[~associated_mask]

            # TODO: Add step
            self.add(new_points, new_normals, new_colors, R, t)
            # End of TODO

            added_entries = len(new_points)
            print('updated: {}, added: {}, total: {}'.format(
                updated_entries, added_entries, len(self.points)))

    def fuse_bonus(self,
                   vertex_map,
                   normal_map,
                   color_map,
                   intrinsic,
                   T,
                   timestamp,
                   dist_diff=0.03,
                   angle_diff=np.deg2rad(5),
                   c_stable=10,
                   t_max=20,
                   static_mask=None,
                   depth_mask=None):
        '''
        \param self The current maintained map
        \param vertex_map Input vertex map, (H, W, 3)
        \param normal_map Input normal map, (H, W, 3)
        \param intrinsic Intrinsic matrix, (3, 3)
        \param T transformation from camera (input) to world (map), (4, 4)
        \param timestamp the current timestamp
        \param dist_diff distance threshold to filter projective association
        \param angle_diff angle threshold to filter projective association
        \param c_stable threshold to determine whether points are stable
        \param t_max threshold to determine whether points are dormant
        \return None, update map properties on demand
        '''
        # Camera to world
        R = T[:3, :3]
        t = T[:3, 3:]

        # World to camera
        T_inv = np.linalg.inv(T)
        R_inv = T_inv[:3, :3]
        t_inv = T_inv[:3, 3:]

        if not self.initialized:
            print("Initializing point clouds...")
            if static_mask is None:
                points = vertex_map.reshape((-1, 3))[depth_mask]
                normals = normal_map.reshape((-1, 3))[depth_mask]
                colors = color_map.reshape((-1, 3))[depth_mask]
            else:
                points = vertex_map.reshape((-1, 3))[static_mask & depth_mask]
                normals = normal_map.reshape((-1, 3))[static_mask & depth_mask]
                colors = color_map.reshape((-1, 3))[static_mask & depth_mask]

            self.add(points, normals, colors, R, t, timestamp)
            self.initialized = True

        else:
            h, w, _ = vertex_map.shape

            # Transform from world to camera for projective association
            indices = np.arange(len(self.points)).astype(int)
            T_points = (R_inv @ self.points.T + t_inv).T
            R_normals = (R_inv @ self.normals.T).T

            # Projective association
            us, vs, ds = transforms.project(T_points, intrinsic)
            us = np.round(us).astype(int)
            vs = np.round(vs).astype(int)

            # first filter: valid projection
            mask = self.filter_pass1(us, vs, ds, h, w)
            # Should not happen -- placeholder before implementation
            if mask.sum() == 0:
                return

            indices = indices[mask]
            us = us[mask]
            vs = vs[mask]

            # second filter: remove points with invalid depth
            print("Before removing invalid depth points: %d matched points" % len(indices))
            _vs, _us = np.unravel_index(np.where(depth_mask), (h, w))
            pxs, mask, _ = transforms.pixel_intersection(
                        np.hstack((vs.reshape(-1,1), us.reshape(-1,1))),
                        np.hstack((_vs.reshape(-1,1), _us.reshape(-1,1))))
            indices = indices[mask]
            us = us[mask]
            vs = vs[mask]
            print("After removing invalid depth points: %d matched points" % len(indices))

            T_points = T_points[indices]
            R_normals = R_normals[indices]
            valid_points = vertex_map[vs, us]
            valid_normals = normal_map[vs, us]

            # third filter: apply thresholds
            mask = self.filter_pass2(T_points, R_normals, valid_points,
                                     valid_normals, dist_diff, angle_diff)
            # Should not happen -- placeholder before implementation
            if mask.sum() == 0:
                return

            indices = indices[mask]
            us = us[mask]
            vs = vs[mask]

            # fourth filter: remove dynamic points
            if static_mask is not None:
                print("Before removing dynamic points: %d matched points" % len(indices))
                _vs, _us = np.unravel_index(np.where(static_mask), (h, w))
                pxs, mask, _ = transforms.pixel_intersection(
                            np.hstack((vs.reshape(-1,1), us.reshape(-1,1))),
                            np.hstack((_vs.reshape(-1,1), _us.reshape(-1,1))))
                indices = indices[mask]
                us = us[mask]
                vs = vs[mask]
                print("After removing dynamic points: %d matched points" % len(indices))

            updated_entries = len(indices)

            merged_points = vertex_map[vs, us]
            merged_normals = normal_map[vs, us]
            merged_colors = color_map[vs, us]

            # Merge step - compute weight average after transformation
            self.merge(indices, merged_points, merged_normals, merged_colors,
                       R, t, timestamp)

            associated_mask = np.zeros((h, w)).astype(bool)
            associated_mask[vs, us] = True
            new_points = vertex_map[~associated_mask]
            new_normals = normal_map[~associated_mask]
            new_colors = color_map[~associated_mask]

            # Add step
            self.add(new_points, new_normals, new_colors, R, t, timestamp)
            
            # Remove step
            self.remove(c_stable, t_max, timestamp)

            added_entries = len(new_points)
            print('updated: {}, added: {}, total: {}'.format(
                updated_entries, added_entries, len(self.points)))

        # Render point clouds back to image to input TartanVO
        img = np.zeros_like(vertex_map)
        T_points = (R_inv @ self.points.T + t_inv).T
        R_normals = (R_inv @ self.normals.T).T
        us, vs, ds = transforms.project(T_points, intrinsic)
        us = np.round(us).astype(int)
        vs = np.round(vs).astype(int)
        img[vs, us] = self.colors[:,::-1]
        img = (img * 256).astype(np.uint8)
        
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', help='path to the dataset folder containing rgb/ and depth/')
    parser.add_argument('--start_idx',
                        type=int,
                        help='index to the source depth/normal maps',
                        default=1)
    parser.add_argument('--end_idx',
                        type=int,
                        help='index to the target depth/normal maps',
                        default=200)
    parser.add_argument('--downsample_factor', type=int, default=2)
    args = parser.parse_args()

    intrinsic_struct = o3d.io.read_pinhole_camera_intrinsic('intrinsics.json')
    intrinsic = np.array(intrinsic_struct.intrinsic_matrix)
    indices, gt_poses = load_gt_poses(
        os.path.join(args.path, 'livingRoom2.gt.freiburg'))
    # TUM convention
    depth_scale = 5000.0

    rgb_path = os.path.join(args.path, 'rgb')
    depth_path = os.path.join(args.path, 'depth')
    normal_path = os.path.join(args.path, 'normal')

    m = Map()

    down_factor = args.downsample_factor
    intrinsic /= down_factor
    intrinsic[2, 2] = 1

    for i in range(args.start_idx, args.end_idx + 1):
        print('Fusing frame {:03d}'.format(i))
        source_depth = o3d.io.read_image('{}/{}.png'.format(depth_path, i))
        source_depth = np.asarray(source_depth) / depth_scale
        source_depth = source_depth[::down_factor, ::down_factor]
        source_vertex_map = transforms.unproject(source_depth, intrinsic)

        source_color_map = np.asarray(
            o3d.io.read_image('{}/{}.png'.format(rgb_path,
                                                 i))).astype(float) / 255.0
        source_color_map = source_color_map[::down_factor, ::down_factor]

        source_normal_map = np.load('{}/{}.npy'.format(normal_path, i))
        source_normal_map = source_normal_map[::down_factor, ::down_factor]

        m.fuse(source_vertex_map, source_normal_map, source_color_map,
               intrinsic, gt_poses[i])
        # m.fuse_bonus(source_vertex_map, source_normal_map, source_color_map,
        #        intrinsic, gt_poses[i], i)

    global_pcd = o3d_utility.make_point_cloud(m.points,
                                              colors=m.colors,
                                              normals=m.normals)
    o3d.visualization.draw_geometries(
        [global_pcd.transform(o3d_utility.flip_transform)])
