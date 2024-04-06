import os
import cv2
import json

import itertools
import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from detectron2.structures import BoxMode

from visualize import InstanceVisualizer, SemanticVisualizer
from .plot import calc_cm, draw_cm
from .metrics import eval_metrics



CLASS_NAMES = ['head', 'neck', 'torso', 'upper_arm', 'lower_arm', 'hand',
                             'hip', 'perineum', 'thigh', 'lower_leg', 'foot']

class SemanticEvaluator:
    """
    Evaluate semantic segmentation metrics.
    """

    gts, preds, scores = [], [], []

    def __init__(self, 
                 output_dir: str=None, 
                 visualize: bool=False, 
                 alpha=0.5, 
                 num_classes=4, 
                 only_mask=False, 
                 save_metrics=False):
        super().__init__()
        self.visualize = visualize
        if output_dir is None:
            output_dir = f"{os.getcwd()}/result"
        self.output_dir = f"{output_dir}/sem_viz"
        os.makedirs(self.output_dir, exist_ok=True)
        self.alpha = alpha
        self.num_classes = num_classes
        self.only_mask = only_mask
        self.save_metrics = save_metrics

    def process(self, inputs, preds):
        for input_, pred in zip(inputs, preds):
            score = pred["sem_seg"].softmax(dim=0).detach(
            ).cpu().numpy() # save to plot p-r curve
            pred = pred["sem_seg"].argmax(dim=0).detach(
            ).cpu().numpy()  # logits -> pred label
            # read ground truth file
            gt = cv2.imread(input_["sem_gt_file"], -1)
            if self.visualize:
                img_name = input_['file_name'].replace(
                    'image', 'image_blur')  # 替换成打码后的图片
                image = cv2.imread(img_name)[..., ::-1]  # bgr -> rgb
                name = img_name.split('/')[-1]

                if self.only_mask:
                    visulizer = SemanticVisualizer(image=np.zeros_like(image), alpha=1)
                else:
                    visulizer = SemanticVisualizer(image=image, alpha=self.alpha)
                # 绘制掩码
                viz_pred = visulizer.visualize_mask(pred)
                viz_gt = visulizer.visualize_mask(gt)

                # 保存可视化绘制结果
                cv2.imwrite(f"{self.output_dir}/{name}",
                            np.hstack([image[..., ::-1], viz_gt[..., ::-1], viz_pred[..., ::-1]]))
            
            if abs(np.unique(gt).size - np.unique(pred).size) <= 1:
                self.preds.append(pred)
                self.gts.append(gt)
                self.scores.append(score)
    
    def reset(self):
        self.preds = []
        self.gts = []
        self.scores = []

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.
        """
        metrics = eval_metrics(self.preds, self.gts, num_classes=self.num_classes, ignore_index=-1)
        if self.save_metrics:
            with open(f"{self.output_dir}/_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=4)
            # confusion matrix
            cm = calc_cm(self.gts, self.preds, self.num_classes)
            draw_cm(cm, num_classes=self.num_classes, sav_dir=self.output_dir)
        return metrics


class InstanceEvaluator:
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation

    """
    predictions, img_ids = [], []
    sem_masks, sem_gts = [], []

    def __init__(
        self,
        output_dir: str = None,
        alpha: float = 0.5,
        visualize: bool = False,
        only_mask: bool = False,
        save_metrics: str = False,
        gt_json_file: str = 'dataset/burns/part_coco/annotations.json',
        class_names: list = CLASS_NAMES,
        text_color: str = 'black',
        viz_box: bool = True
    ):  
        if output_dir is None:
            output_dir = f"{os.getcwd()}/result"
        self.output_dir = f"{output_dir}/ist_viz"
        os.makedirs(self.output_dir, exist_ok=True)
        self.alpha = alpha
        self.visualize = visualize
        self.only_mask = only_mask
        self.save_metrics = save_metrics
        self.gt_json_file = gt_json_file
        self.class_names = class_names
        self.text_color = text_color
        self.viz_box = viz_box

        self.metadata = {"thing_classes": class_names}

    def reset(self):
        self.predictions = []
        self.img_ids = []
        self.sem_masks = []
        self.sem_gts = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input_, output in zip(inputs, outputs):
            image_id = input_['image_id']
            self.img_ids.append(image_id)

            prediction = {"image_id": image_id}
            if "instances" in output:
                instances = output["instances"].to('cpu')
                prediction["instances"] = predictions_to_coco_json(
                    instances, image_id)
                
                # for calculating dice
                # NxHxW
                pred_masks = instances.pred_masks
                pred_classes = instances.pred_classes
                N, H, W = pred_masks.shape
                sem_mask = np.zeros((H, W), dtype=np.uint8)
                for i in range(N):
                    sem_mask[pred_masks[i]] = pred_classes[i] + 1
                sem_gt = cv2.imread(input_['file_name'].replace(
                    'image', 'part_coco/label').replace('jpg', 'png'), 0)
                self.sem_masks.append(sem_mask)
                self.sem_gts.append(sem_gt)
                
                if self.visualize:
                    img_name = input_['file_name'].replace('image', 'image_blur')
                    image = cv2.imread(img_name)[..., ::-1]  # bgr -> rgb
                    name = img_name.split('/')[-1]
                    if self.only_mask:
                        visualizer = InstanceVisualizer(
                            img_rgb=np.zeros_like(image), metadata=self.metadata,
                            viz_box=self.viz_box, text_align='center', alpha=self.alpha, 
                            text_color=self.text_color)
                    else:
                        visualizer = InstanceVisualizer(
                            img_rgb=image, metadata=self.metadata,
                            viz_box=self.viz_box, text_align='center', alpha=self.alpha,
                            text_color=self.text_color)
                    gt = cv2.imread(input_['file_name'].replace(
                        'image', 'part_coco/viz').replace('jpg', 'png'))  # 读取可视化ground truth
                    viz = visualizer.draw_instance_predictions(instances)
                    viz = viz.get_image()[...,::-1]
                    cv2.imwrite(f"{self.output_dir}/{name}",
                                np.hstack([image[..., ::-1], gt, viz]))

            if len(prediction) > 1:
                self.predictions.append(prediction)

    def evaluate(self):
        # # doesn't have any instances, return empty dict
        # if not self.predictions[0].get('instances'):
        #     return {'AP': 0, 'AP50': 0, 'AP75': 0}
        # predictions and ground truthes
        coco_gt = COCO(self.gt_json_file)
        coco_results = list(itertools.chain(
            *[x["instances"] for x in self.predictions]))
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
        # only reserve image ids in test dataset
        coco_eval.params.imgIds = self.img_ids  # ！！！！
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        #
        metrics = self._derive_coco_results(
            coco_eval, 'segm', self.class_names)
        # calculate dice
        if len(self.sem_masks) > 0:
            cm = calc_cm(self.sem_gts, self.sem_masks, len(self.class_names) + 1)
            # print(cm)
            draw_cm(cm, num_classes=len(self.class_names) + 1, sav_dir=self.output_dir)
            metrics.update(eval_metrics(self.sem_masks, self.sem_gts,
                                        num_classes=len(self.class_names) + 1,
                                        ignore_index=-1))
        if self.save_metrics:
            with open(f"{self.output_dir}/_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=4)
        return metrics

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75"],
            "segm": ["AP", "AP50", "AP75"],
        }[iou_type]

        # the standard metrics
        results = {
            metric: float(
                (coco_eval.stats[idx] * 100).round(2) if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        # Compute per-category AP
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(
                precision * 100).round(2) if precision.size else float("nan")
            results_per_category.append((f"{name}", ap))
        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results

def predictions_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(
                np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        results.append(result)
    return results
