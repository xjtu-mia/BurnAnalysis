import torch
import numpy as np
from collections import OrderedDict


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):

    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label, total_area = total_intersect_and_union(
            results, gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label)
    ret_metrics = total_counting_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label, total_area_label, 
                                        total_area, beta)
    return ret_metrics

def f_score(precision, recall, beta=1):
    """calculate the f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score

def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map
        label (ndarray): Ground truth segmentation map
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    pred_label = torch.from_numpy((pred_label))
    label = torch.from_numpy(label)

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    valid_mask = (label != ignore_index)
    pred_label = pred_label[valid_mask]
    label = label[valid_mask]

    intersect = pred_label[pred_label == label]
    # torch.histc: computes the histogram of a tensor
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label

def total_intersect_and_union(results: torch.Tensor,
                              gt_seg_maps: torch.Tensor,
                              num_classes: int,
                              ignore_index: int,
                              label_map=dict(),
                              reduce_zero_label: bool=False):
    """Calculate Total Intersection and Union.
    """
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area = torch.zeros((num_classes, ), dtype=torch.float64)

    for result, gt_seg_map in zip(results, gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
        # 统计所有像素
        total_area += torch.tensor([gt_seg_map.size] * num_classes, dtype=torch.int64)    
    return total_area_intersect, total_area_union, total_area_pred_label, total_area_label, total_area

def total_counting_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          total_area,
                          beta=1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        total_area (ndarray): Total pixels of ground truth, shape 
            (num_classes, ).
     Returns:
        Overall metrics on all images, OrderDict(str, list).
        metric (list): Per category evaluation metrics, shape (num_classes, ).
    """
    # 总体准确率 sum(tp) / sum(tp+fn)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc.reshape(1)})
    # iou == jaccard index  对于分割任务 计算公式为   tp/tp+fp+fn
    iou = total_area_intersect / total_area_union
    acc = total_area_intersect / total_area_label # Tp / Tp + Fn
    ret_metrics['IoU'] = iou
    ret_metrics['Acc'] = acc
    # dice == f1-score  
    # 灵敏度(sensitivity = tp/tp+fn, 又名真阳性率,计算公式与召回率相同), 特异度(specificity = tn/tn+fp, 真阴性率)
    dice = 2 * total_area_intersect / (
        total_area_pred_label + total_area_label) # 2tp/tp+fp+tp+fn
    ret_metrics['Dice'] = dice
    # f1-score是精确度(precision = tp/tp+fp,又名查准率)与召回率(recall = tp/tp+fn查全率)的调和平均数
    precision = total_area_intersect / total_area_pred_label # tp/tp+fp
    recall = total_area_intersect / total_area_label # tp/tp+fn
    f_value = torch.tensor(
        [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
    ret_metrics['Precision'] = precision
    ret_metrics['Recall'] = recall
    ret_metrics[f'F{beta}score'] = f_value
    # 灵敏度 真阳性率
    ret_metrics['Sensitivity'] = recall # tp/tp+fn=recall
    # 特异性 真阴性率 specificity = tn/tn+fp=(total-union)/(total-total-gt) 
    ret_metrics['Specificity'] = (total_area - total_area_union) / (total_area - total_area_label) 

    ret_metrics = {
        name: [(i * 100).round(2) for i in value.numpy()] # 保留四舍五入两位小数
        for name, value in ret_metrics.items()
    }
    return ret_metrics
