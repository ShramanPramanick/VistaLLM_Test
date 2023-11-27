SEG_POINTS = 40
MASKS_POINTS = 16
SCALE = 1000

import sys
import logging
import warnings
import numpy
import math
from typing import Dict, Any, Sequence
import numpy as np
import cv2

import torch
from torchvision.ops import box_iou

from ..utils import (
    MInstrDataset,
    BaseComputeMetrics,
)

from ..process_function import (
    BoxFormatter,
)

from ..root import (
    DATASETS,
    METRICS,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    EXPR_PLACEHOLDER,
    MASKS_PLACEHOLDER
)

def contour2mask(target):
  target_x = target[0:-1:2].unsqueeze(-1)
  target_y = target[1::2].unsqueeze(-1)
  target = tuple([torch.cat((target_x, target_y), dim=-1).unsqueeze(1).numpy()])
  return target

def iou(mask1, mask2):
    intersection = (mask1 & mask2).sum()
    if intersection == 0:
        return 0.0
    union = (mask1 | mask2).sum()
    return intersection / union




logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

def convert_mask_anno(contour):
    contour = numpy.array(contour)
    num_pts = contour.shape[0]
    if SEG_POINTS is not None:
        vertices = numpy.empty((1, SEG_POINTS*2), dtype=numpy.float32)
    else:
        vertices = numpy.empty((1, MASKS_POINTS*2), dtype=numpy.float32)
    vertices.fill(-1)
    if SEG_POINTS is not None:
        interval = math.ceil(num_pts / SEG_POINTS) ## math.ceil
    else:
        interval = math.ceil(num_pts / MASKS_POINTS) ## math.ceil
    vertex_x = contour[::interval, 0]
    vertex_y = contour[::interval, 1]
    partial_vertices = numpy.vstack((vertex_x, vertex_y))  # 2 x num_vertices
    # vertices[:, :partial_vertices.shape[1]] = partial_vertices
    vertices[0, 0 : partial_vertices.shape[1]*2-1 : 2] = partial_vertices[0]
    vertices[0, 1 : partial_vertices.shape[1]*2 : 2] = partial_vertices[1]
    return vertices, partial_vertices

def compute_breaks(points, n=1):
    break_list = []
    slope_diff = []
    for i in range(n):
        break_list.append(points[i,:])
        slope_diff.append(180.)
    for i in range(n, len(points)-n):
        if points[i,0] == points[i,1] == -1:
            break_list.append(points[i-1,:])
            slope_diff.append(180.)
            return break_list, np.asarray(slope_diff)
        else:
            p1 = points[i-n,:]
            p2 = points[i,:]
            p3 = points[i+n,:]
            break_list.append(points[i,:])
            slope_diff.append(np.abs(np.cross(p3-p1, p1-p2)) / np.linalg.norm(p3-p1))
    for i in range(len(points)-n, len(points)):
        break_list.append(points[i,:])
        slope_diff.append(180.)

    return break_list, np.asarray(slope_diff)



def adaptive_sampling(points, final_pts=80, n=1):
    break_list, slope_diff = compute_breaks(points, n=n)
    sorted_args = np.argsort(slope_diff)

    remove_pts = len(slope_diff) - final_pts
    remove_sorted_args = sorted_args[:remove_pts]
    remove_sorted_dict = {}

    for i in range(len(remove_sorted_args)):
        remove_sorted_dict.update({str(remove_sorted_args[i]):True})

    break_list_new = []

    for i in range(len(break_list)):
        if str(i) not in remove_sorted_dict:
            break_list_new.append(break_list[i])

    return np.asarray(break_list_new)


@DATASETS.register_module()
class RESDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['img_path']
        expr = item['expression']
        mask, _ = convert_mask_anno(item['mask'])
        mask = mask[0]

        if SEG_POINTS is not None:
            _, mask = convert_mask_anno(item['mask'])
            break_list = adaptive_sampling(mask.T, final_pts=MASKS_POINTS, n=1) ## make this a variable
            mask = break_list.T
            vertices = numpy.empty((1, MASKS_POINTS*2), dtype=numpy.float32)
            vertices.fill(-1)
            vertices[0, 0 : mask.shape[1]*2-1 : 2] = mask[0]
            vertices[0, 1 : mask.shape[1]*2 : 2] = mask[1]
            mask = vertices[0]

        # print("mask.shape", mask.shape)
        # print(abcd)

        # print("img_path: ", img_path)
        # print("expr: ", expr)
        # print("mask: ", mask)
        # print("height: ", item['height'])
        # print("width: ", item['width'])
        # print(abcd)

        image = self.get_image(img_path)
        question = self.get_template().replace(EXPR_PLACEHOLDER, expr)

        ret = {
            'image': image,
            'target': {
                'masks': [mask],
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f'Answer: {MASKS_PLACEHOLDER} .',
                    'masks_seq': [[0]],
                }
            ]
        }
        return ret


@METRICS.register_module()
class RESComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_formatter: BoxFormatter = self.preprocessor['target']['boxes']

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        failed = 0
        target_failed = 0

        pred_boxes, target_boxes = [], []
        for pred, target in zip(preds, targets):
            # print("pred_res: ", pred)
            # print("target: ", target)
            # import pdb
            # pdb.set_trace()
            extract_pred = self.extract_ans(pred)
            extract_target = self.extract_ans(target)
            if extract_target is None:
                target_failed += 1
                logger.warning(f"failed to extract ans for target: {target}")
                continue
            if extract_pred is None:
                failed += 1
                logger.warning(f"failed to extract ans for pred: {pred}")
                extract_pred = [0 for idx in range(MASKS_POINTS*2)]
            target_boxes.append(extract_target)
            pred_boxes.append(extract_pred)


        mIoU_list = []
        with torch.no_grad():
            target_boxes = torch.tensor(target_boxes)
            pred_boxes = torch.tensor(pred_boxes)
            for idx in range(len(target_boxes)):
                # normalized box value is too small, so that the area is 0.
                target = target_boxes[idx]
                pred = pred_boxes[idx]
                target = (target*SCALE).to(torch.long)
                pred = (pred*SCALE).to(torch.long)
                target = contour2mask(target)
                pred = contour2mask(pred)

                mask_target = np.zeros((SCALE, SCALE), np.uint8)
                mask_target = cv2.drawContours(mask_target, target, 0, 255, -1)
                mask_pred = np.zeros((SCALE, SCALE), np.uint8)
                mask_pred = cv2.drawContours(mask_pred, pred, 0, 255, -1)

                mask_iou = torch.tensor([-1.])
                # mask_acc_at_thrs = torch.full((5, ), -1.)
                mask_iou = iou(mask_target, mask_pred)
                #     for i, iou_thr in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
                #         mask_acc_at_thrs[i] = (
                #             mask_iou >= iou_thr).mean()
                mIoU_list.append(mask_iou)

        # HACK: currently we expand image to square. so this iou is the real iou.
        warn_message = "this iou is calculate on normalized box. just for non-rigorous training progress checking." \
                       "the value is consistent with real iou only if image.width == image.height."
        warnings.warn(warn_message)
        from statistics import mean

        return {
            'mIoU': mean(mIoU_list),
            'target_failed': target_failed,
            'failed': failed,
            'warning': warn_message,
        }

    def extract_ans(self, string: str):
        try:
            list_of_boxes = self.box_formatter.extract_mask(string)
            if len(list_of_boxes) != 1 or len(list_of_boxes[0]) != 1:
                return None
            box = list_of_boxes[0][0]
            if len(box) != MASKS_POINTS*2:
                return None
            return box
        except Exception as e:
            logger.warning(f"extract_ans for {string} but get exception: {e}")
            return None
