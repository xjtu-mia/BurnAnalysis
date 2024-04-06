import torch
from torch.nn import functional as F

from detectron2.layers import batched_nms
from detectron2.structures import Instances, ROIMasks, Boxes


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False)[0]
    return result


# perhaps should rename to "resize_instance"
def detector_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5
):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(new_size)

    model_name = results._model_name
    results = results[output_boxes.nonempty()]

    scores = results.get('scores')
    pred_masks = results.get('pred_masks')
    pred_classes = results.get('pred_classes')
    pred_boxes = results.get('pred_boxes').tensor

    # class agnostic non maximum suppression
    keep = batched_nms(pred_boxes, scores, idxs=torch.ones_like(scores), iou_threshold=0.5)
    results = Instances(new_size)
    results.pred_boxes = Boxes(pred_boxes[keep])
    results.pred_masks = pred_masks[keep]
    results.pred_classes = pred_classes[keep]
    results.scores = scores[keep]

    # maskrcnn 
    if model_name in ['maskrcnn', 'pointrend']:
        roi_masks = results.pred_masks # pred_masks is a tensor of shape (N, 1, M, M)
        roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
        pred_masks = roi_masks.to_bitmasks(
            results.pred_boxes, output_height, output_width, mask_threshold).tensor 
    # solo, sparseinst 
    elif model_name in ['solo', 'sparseinst', 'maskdino']:
        pred_masks = results.pred_masks
        pred_masks = F.interpolate(pred_masks.unsqueeze(0), size=(output_height, output_width), mode='bilinear')[0]
        pred_masks = pred_masks > mask_threshold

    # 消除不同实例的mask之间的混叠
    # pred_masks = eliminate_overlap(pred_masks)
    results.pred_masks = pred_masks

    return results

def eliminate_overlap(mask_tensors):
    num_masks = mask_tensors.shape[0]
    if num_masks <= 1:
        return mask_tensors
    else:
        for i in range(num_masks):
            for j in range(i + 1, num_masks):
                mask1 = mask_tensors[i]
                mask2 = mask_tensors[j]
                inter = mask1 * mask2
                mask1 = mask1 ^ inter
                mask_tensors[i] = mask1
        return mask_tensors