import sys
sys.path.append('./evaluator')

from torch.utils.data import DataLoader
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow, load_kiiti_intrinsics, tensor2img
from Datasets.tartanTrajFlowDataset import TrajFolderDataset
from Datasets.transformation import ses2poses_quat
from evaluator.tartanair_evaluator import TartanAirEvaluator
from TartanVO import TartanVO

import argparse
import numpy as np
import cv2
import torch
from os import mkdir
from os.path import isdir

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')
    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 448)')
    parser.add_argument('--model-name', default='',
                        help='name of pretrained model (default: "")')
    parser.add_argument('--euroc', action='store_true', default=False,
                        help='euroc test (default: False)')
    parser.add_argument('--fr3', action='store_true', default=False,
                        help='fr3 test (default: False)')
    parser.add_argument('--kitti', action='store_true', default=False,
                        help='kitti test (default: False)')
    parser.add_argument('--kitti-intrinsics-file',  default='',
                        help='kitti intrinsics file calib.txt (default: )')
    parser.add_argument('--test-dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--pose-file', default='',
                        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")')
    parser.add_argument('--save-flow', action='store_true', default=False,
                        help='save optical flow (default: False)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    testvo = TartanVO(args.model_name)

    # load trajectory data from a folder
    datastr = 'tartanair'
    if args.kitti:
        datastr = 'kitti'
    elif args.euroc:
        datastr = 'euroc'
    elif args.fr3:
        datastr = 'fr3'
    else:
        datastr = 'tartanair'
    focalx, focaly, centerx, centery = dataset_intrinsics(datastr) 
    if args.kitti_intrinsics_file.endswith('.txt') and datastr=='kitti':
        focalx, focaly, centerx, centery = load_kiiti_intrinsics(args.kitti_intrinsics_file)

    transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])

    testDataset = TrajFolderDataset(args.test_dir,  posefile = args.pose_file, transform=transform, 
                                        focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)
    testDataloader = DataLoader(testDataset, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=args.worker_num)
    testDataiter = iter(testDataloader)

    motionlist = []
    testname = datastr + '_' + args.model_name.split('.')[0]
    if args.save_flow:
        flowdir = 'results/'+testname+'_flow'
        if not isdir(flowdir):
            mkdir(flowdir)
        flowcount = 0

    for i, sample in enumerate(testDataloader):

        motions, flow = testvo.test_batch(sample)
        
        # iteration to refine pose estimate
        for _ in range(1):
            '''
            TODO: Integrate sceneflow.py --> mask of the static scene
            This iteration is to refine pose estimation by iteratatively removing 
            dynamic pixels from the image via thresholding either (1) 2D scene flow
            or (2) geometric distance to epipolar lines.
            '''
            # Mask and Inpaint (inpaint makes pose estimation even worse, 
            # and significantly increases runtime)
            mask1 = np.load('data/fr3_walking_xyz/cropped/mask/' + str(i).zfill(4) + '_mask.npy')
            rgb1 = tensor2img(sample['img1'][0]).copy()
            rgb1 = cv2.bitwise_and(rgb1, rgb1, mask=mask1.astype(np.uint8))
            # res1 = rgb1.copy()
            # cv2.xphoto.inpaint(rgb1, mask1.astype(np.uint8), res1, cv2.xphoto.INPAINT_FSR_FAST)
            sample['img1'][0] = torch.from_numpy(rgb1.transpose(2,0,1) / 255.0)

            mask2 = np.load('data/fr3_walking_xyz/cropped/mask/' + str(i+1).zfill(4) + '_mask.npy')
            rgb2 = tensor2img(sample['img2'][0]).copy()
            rgb2 = cv2.bitwise_and(rgb2, rgb2, mask=mask2.astype(np.uint8))
            # res2 = rgb2.copy()
            # cv2.xphoto.inpaint(rgb2, mask2.astype(np.uint8), res2, cv2.xphoto.INPAINT_FSR_FAST)
            sample['img2'][0] = torch.from_numpy(rgb2.transpose(2,0,1) / 255.0)

            motions, flow = testvo.test_batch(sample)
        
        motionlist.extend(motions)

        if args.save_flow:
            for k in range(flow.shape[0]):
                flowk = flow[k].transpose(1,2,0)
                np.save(flowdir+'/'+str(flowcount).zfill(6)+'.npy',flowk)
                flow_vis = visflow(flowk)
                cv2.imwrite(flowdir+'/'+str(flowcount).zfill(6)+'.png',flow_vis)
                flowcount += 1

    poselist = ses2poses_quat(np.array(motionlist))

    # calculate ATE, RPE, KITTI-RPE
    if args.pose_file.endswith('.txt'):
        evaluator = TartanAirEvaluator()
        results = evaluator.evaluate_one_trajectory(args.pose_file, poselist, scale=True, kittitype=(datastr=='kitti'))
        if datastr=='euroc' or 'fr3':
            print("==> ATE: %.4f" %(results['ate_score']))
        else:
            print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))

        # save results and visualization
        plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results/'+testname+'.png', title='ATE %.4f' %(results['ate_score']))
        np.savetxt('results/'+testname+'.txt',results['est_aligned'])
    else:
        np.savetxt('results/'+testname+'.txt',poselist)