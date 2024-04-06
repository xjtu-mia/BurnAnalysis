"""
Implemention Mask RCNN base on offical code. 
Ref: https://github.com/facebookresearch/detectron2
Paper: https://arxiv.org/pdf/1703.06870.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from detectron2.modeling.proposal_generator import StandardRPNHead, RPN
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.modeling.roi_heads.box_head import FastRCNNConvFCHead
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.mask_head import MaskRCNNConvUpsampleHead
from detectron2.layers import ShapeSpec
from detectron2.structures import ImageList, Instances


class MaskRCNNHead(nn.Module):
    def __init__(self, num_classes: int = 11):
        """
        NOTE: 需要注释掉 rpn, roi_heads 中的get_event_storage()相关代码段(训练过程中日志保存相关代码)
        """
        super().__init__()
        # -----------build region proposal netwoek-----------
        anchor_generator = DefaultAnchorGenerator(
            sizes=[[32], [64], [128], [256]],
            aspect_ratios=[[0.5, 1, 2]],
            strides=[4, 8, 16, 32],
            offset=0.5
        )
        rpn_head = StandardRPNHead(in_channels=256,
                                   num_anchors=anchor_generator.num_anchors[0],
                                   box_dim=4,
                                   conv_dims=[256, 256])
        matcher = Matcher(thresholds=[0.3, 0.7], labels=[
                          0, -1, 1], allow_low_quality_matches=True)
        box2box_transform = Box2BoxTransform(weights=[1.0, 1.0, 1.0, 1.0])
        self.rpn = RPN(
            in_features=['p2', 'p3', 'p4', 'p5'],
            head=rpn_head,
            anchor_generator=anchor_generator,
            anchor_matcher=matcher,
            box2box_transform=box2box_transform,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_topk=[2000, 1500],
            post_nms_topk=[1000, 750]
        )
        # -----------build Roi subnets for box, class, mask prediction-------------------
        box_pooler = ROIPooler(
            output_size=7,
            scales=tuple(1.0 / stride for stride in [4, 8, 16, 32]),
            sampling_ratio=0,
            pooler_type='ROIAlignV2',
        )
        box_head = FastRCNNConvFCHead(input_shape=ShapeSpec(channels=256, height=7, width=7),
                                      conv_dims=[256],
                                      fc_dims=[1024, 1024])
        box_predictor = FastRCNNOutputLayers(
            input_shape=box_head.output_shape,
            box2box_transform=box2box_transform,
            num_classes=num_classes,
            test_score_thresh=0.5,
            test_nms_thresh=0.5,
            test_topk_per_image=128
        )
        mask_pooler = ROIPooler(
            output_size=14,
            scales=tuple(1.0 / stride for stride in [4, 8, 16, 32]),
            sampling_ratio=0,
            pooler_type='ROIAlignV2',
        )
        mask_head = MaskRCNNConvUpsampleHead(
            ShapeSpec(channels=256, height=14, width=14),
            num_classes=num_classes,
            conv_dims=[256, 256, 256, 256]
        )
        self.roi_heads = StandardROIHeads(
            box_in_features=['p2', 'p3', 'p4', 'p5'],
            box_pooler=box_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            mask_in_features=['p2', 'p3', 'p4', 'p5'],
            mask_pooler=mask_pooler,
            mask_head=mask_head,
            num_classes=num_classes,
            batch_size_per_image=128,
            proposal_matcher=Matcher(thresholds=[0.5], labels=[0, 1],
                                     allow_low_quality_matches=False),
            positive_fraction=0.5,
            proposal_append_gt=True
        )

    def forward(self, 
                features: Dict[str, torch.Tensor],
                images: ImageList,
                gt_instances: List[Instances]):
        """
        Args: 
        features: features of FPN
        gt_instances: ground truth
        """
        losses = {}
        # rpn loss
        proposals, proposal_losses = self.rpn(images, features, gt_instances)
        losses.update(proposal_losses)
        # roi_heads loss
        if not self.training and gt_instances is not None:
            _, detector_losses = self.roi_heads.val(
            images, features, proposals, gt_instances)
        else:
            _, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances)
        losses.update(detector_losses)
        return losses

    def inference(self, 
                features: Dict[str, torch.Tensor],
                images: ImageList):
        # predict
        proposals, _ = self.rpn(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)
        tag_results = []
        for result in results:
            result._model_name = 'maskrcnn'
            tag_results.append(result)
        return tag_results
