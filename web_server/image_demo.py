import os
from argparse import ArgumentParser

import cv2
import torch

from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)


def parse_args():
    parser = ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--img_root', type=str, default='', help='Image root')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.01, help='Keypoint score threshold')
    args = parser.parse_args()
    return args

def get_pose(img, result_path, pose_config='./mobilenetv2_coco_512x512.py',
             pose_checkpoint='./mobilenetv2_coco_512x512-4d96e309_20200816.pth', device='cpu', kpt_thr=0.5):

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        pose_config, pose_checkpoint, device=device.lower())
    # optional
    return_heatmap = False
    dataset = pose_model.cfg.data['test']['type']
    assert (dataset == 'BottomUpCocoDataset')

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    img = cv2.imread(img)

    pose_results, returned_outputs = inference_bottom_up_pose_model(
        pose_model,
        img,
        return_heatmap=return_heatmap,
        outputs=output_layer_names)
    # show the results
    vis_img = vis_pose_result(
        pose_model,
        img,
        pose_results,
        dataset=dataset,
        kpt_score_thr=kpt_thr,
        show=False)
    cv2.imwrite(result_path, vis_img)

    sample0 = {"url": result_path}

    res_list = [sample0]

    return res_list

def main():
    args = parse_args()

    device = torch.device(args.device)

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())
    # optional
    return_heatmap = False
    dataset = pose_model.cfg.data['test']['type']
    assert (dataset == 'BottomUpCocoDataset')

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        # ret_val, img = camera.read()
        img = cv2.imread(args.img_root)

        pose_results, returned_outputs = inference_bottom_up_pose_model(
            pose_model,
            img,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        # show the results
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            dataset=dataset,
            kpt_score_thr=args.kpt_thr,
            show=False)
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        cv2.imshow('Image', vis_img)


if __name__ == '__main__':
    main()
