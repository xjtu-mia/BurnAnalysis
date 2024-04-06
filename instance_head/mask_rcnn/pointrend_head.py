"""
Implemention PointRend base on offical code. 
Ref: https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend
Paper: https://arxiv.org/abs/1912.08193.pdf
"""

import torch
import torch.nn as nn
from typing import Dict, List

from detectron2.config import get_cfg
from detectron2.modeling.proposal_generator import StandardRPNHead, RPN
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.modeling.roi_heads.box_head import FastRCNNConvFCHead
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.layers import ShapeSpec
from detectron2.structures import ImageList
from detectron2.projects.PointRend.point_rend.config import add_pointrend_config
from detectron2.projects.PointRend.point_rend.mask_head import PointRendMaskHead


class RCNNPointRendHead(nn.Module):
    def __init__(self, 
                 num_classes: int = 11,
                 cfg = None,
                 in_features: List[str] = ["p2", "p3", "p4", "p5"],
                 in_channels: List[int] = [256, 256, 256, 256],
                 strides: List[int] = [4, 8, 16, 32],
                 train_num_points: int = 14 * 14,
                 subdivision_steps: int = 5,
                 subdivision_num_points: int = 28 * 28,
                 ):
        super().__init__() 
        # -----------build region proposal netwoek-----------
        anchor_generator = DefaultAnchorGenerator(
            sizes=[[32], [64], [128], [256]],
            aspect_ratios=[[0.5, 1, 2]],
            strides=[4, 8, 16, 32],
            offset=0.5
        )
        rpn_head = StandardRPNHead(in_channels=in_channels[-1],
                                   num_anchors=anchor_generator.num_anchors[0],
                                   box_dim=4,
                                   conv_dims=[in_channels[-1]] * 2)
        matcher = Matcher(thresholds=[0.3, 0.7], labels=[
                          0, -1, 1], allow_low_quality_matches=True)
        box2box_transform = Box2BoxTransform(weights=[1.0, 1.0, 1.0, 1.0])
        self.rpn = RPN(
            in_features=['p2', 'p3', 'p4', 'p5'],
            head=rpn_head,
            anchor_generator=anchor_generator,
            anchor_matcher=matcher,
            box2box_transform=box2box_transform,
            batch_size_per_image=128,
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
        box_head = FastRCNNConvFCHead(input_shape=ShapeSpec(channels=in_channels[-1], height=7, width=7),
                                      conv_dims=[in_channels[-1]],
                                      fc_dims=[1024, 1024])
        box_predictor = FastRCNNOutputLayers(
            input_shape=box_head.output_shape,
            box2box_transform=box2box_transform,
            num_classes=num_classes,
            test_score_thresh=0.5,
            test_nms_thresh=0.5,
            test_topk_per_image=128
        )
        # point mask head ------->
        if cfg == None:
            cfg = get_cfg()
            add_pointrend_config(cfg)
            cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON = True
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
            cfg.MODEL.POINT_HEAD.NUM_CLASSES = num_classes
            cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS = train_num_points
            cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS = subdivision_steps
            # Maximum number of points selected at each subdivision step (N).
            cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS = subdivision_num_points
            cfg.MODEL.ROI_MASK_HEAD.CONV_DIM = in_channels[-1]
        self.in_features = in_features
        
        output_shape = {name : ShapeSpec(channels=channel, stride=stride) 
                        for name, channel, stride in zip(in_features, in_channels, strides)}
        mask_head = PointRendMaskHead(cfg, output_shape)
        # -------------------->
        self.roi_heads = StandardROIHeads(
            box_in_features=['p2', 'p3', 'p4', 'p5'],
            box_pooler=box_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            mask_in_features=['p2', 'p3', 'p4', 'p5'],
            mask_pooler=None,
            mask_head=mask_head,
            num_classes=num_classes,
            batch_size_per_image=128,
            proposal_matcher=Matcher(thresholds=[0.5], labels=[0, 1],
                                     allow_low_quality_matches=False),
            positive_fraction=0.5,
            proposal_append_gt=True
        )

    @torch.jit.unused
    def forward(self, 
                features: Dict[str, torch.Tensor],
                images: ImageList,
                gt_instances: List[Dict[str, torch.Tensor]]):
        """
        Args: 
        features: features dict of FPN
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
            result._model_name = 'pointrend'
            tag_results.append(result)
        return tag_results
    