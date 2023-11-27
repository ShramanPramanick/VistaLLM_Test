import sys
import logging
import warnings
from typing import Dict, Any, Sequence

import torch
from torchvision.ops import generalized_box_iou

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
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


@DATASETS.register_module()
class GRECDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['img_path']
        expr = item['expression']
        bbox = item['bbox']

        image = self.get_image(img_path)
        question = self.get_template().replace(EXPR_PLACEHOLDER, expr)

        if len(bbox) > 0:
            ret = {
            'image': image,
            'target': {
                'boxes': bbox,
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f'Answer: {BOXES_PLACEHOLDER} .',
                    'boxes_seq': [range(0, len(bbox))],
                }
                ]
            }
        
        else:
            ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f' ',
                }
                ]
            }


        return ret


@METRICS.register_module()
class GRECComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_formatter: BoxFormatter = self.preprocessor['target']['boxes']
        self.thresh_iou = 0.5
        self.thresh_F1 = 1.0
    

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        failed = 0
        # target_failed = 0
        correct_image = 0
        notarget = 0
        nt = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

        pred_boxes, target_boxes = [], []
        for pred, target in zip(preds, targets):
            # print("pred_res: ", pred)
            # print("target: ", target)
            extract_pred = self.extract_ans(pred)
            extract_target = self.extract_ans(target)

            if extract_target is None:
                notarget += 1
                if extract_pred is None:
                    if '<b_st>' in pred or '<b_ed>' in pred or '<bin_' in pred:
                        nt["FN"] += 1
                        pass
                    else:
                        nt["TP"] += 1
                        correct_image += 1
                else:
                    nt["FN"] += 1
                continue

            # if extract_target is None:
            #     target_failed += 1
            #     logger.warning(f"failed to extract ans for target: {target}")
            #     continue
            if extract_target is not None:
                if extract_pred is None:
                    nt["FP"] += 1
                    failed += 1
                    logger.warning(f"failed to extract ans for pred: {pred}")
                    extract_pred = [[0, 0, 0, 0]]
                else:
                    nt["TN"] += 1
            
            target_boxes.append(extract_target)
            pred_boxes.append(extract_pred)

        

        with torch.no_grad():
            for (target, pred) in zip(target_boxes, pred_boxes):
                TP = 0
                pred = torch.tensor(pred)
                target = torch.tensor(target)
                cur_giou = generalized_box_iou(pred * 1000, target * 1000)
                num_prediction = pred.shape[0]
                num_gt = target.shape[0]

                for i in range(min(num_prediction, num_gt)):
                    top_value, top_index = torch.topk(cur_giou.flatten(0, 1), 1) ## k = 1 
                    if top_value < self.thresh_iou:
                        break
                    else:
                        top_index_x = top_index // num_gt
                        top_index_y = top_index % num_gt
                        TP += 1
                        cur_giou[top_index_x[0], :] = 0.0
                        cur_giou[:, top_index_y[0]] = 0.0
                    FP = num_prediction - TP
                    FN = num_gt - TP
                    F_1 = 2 * TP / (2 * TP + FP + FN)

                if F_1 >= self.thresh_F1:
                    correct_image += 1

        # HACK: currently we expand image to square. so this iou is the real iou.
        warn_message = "this iou is calculate on normalized box. just for non-rigorous training progress checking." \
                       "the value is consistent with real iou only if image.width == image.height."
        warnings.warn(warn_message)

        return {
            'score': 1.0 * correct_image / len(targets),
            'T_acc': nt['TN'] / (nt['TN'] + nt['FP']),
            'N_acc': nt['TP'] / (nt['TP'] + nt['FN']),
            'notarget_samples': notarget,
            'failed': failed,
            'warning': warn_message,
        }

    def extract_ans(self, string: str):
        try:
            list_of_boxes = self.box_formatter.extract(string)
            if len(list_of_boxes) != 1:  #or len(list_of_boxes[0]) != 1:
                return None
            boxes = list_of_boxes[0]
            for box in boxes:
                if len(box) != 4:
                    return None
            return boxes
        except Exception as e:
            logger.warning(f"extract_ans for {string} but get exception: {e}")
            return None
