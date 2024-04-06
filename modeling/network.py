import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from backbone import *
from backbone.fpn import *
from decode_head import *
from instance_head import *

from detectron2.structures import ImageList, Instances


def select_backbone(name, arch, depth):
    if name == 'swin':
        return SwinTransformer(arch=arch)
    elif name == 'convnext':
        return ConvNeXt(arch=arch)
    elif name == 'resnet':
        return ResNet(depth=depth)
    elif name == 'resnext':
        return ResNeXt(depth=depth)
    elif name == 'mpvit':
        return MPViT(arch=arch)
    elif name == 'maxvit':
        return MaxVit(arch=arch)
    else:
        print(f"Unsupported backbone: {name}")


def select_instance_decoder(name, num_classes):
    if name == 'MaskRCNN':
        return MaskRCNNHead(num_classes)
    elif name == 'Pointrend':
        return RCNNPointRendHead(num_classes, in_channels=[256] * 4)


def select_burn_decoder(name: str, 
                        num_classes: int, 
                        class_weights: List[int],
                        in_channels: List[int]):
    if name == "UNet":
        return UNetHead(num_classes=num_classes, class_weights=class_weights, in_channels=in_channels)
    elif name == "BurnDecode":
        return BurnDecodeHead(num_classes=num_classes, class_weights=class_weights, in_channels=in_channels, 
                          conv_dim=256, up_type='dscu')
    elif name == "BurnDecodeAG":
        return BurnDecodeAGHead(num_classes=num_classes, class_weights=class_weights, in_channels=in_channels, 
                          conv_dim=256, up_type='dscu', guide_in_channels=[256]*4)


class MTNet(nn.Module):
    def __init__(self,
                 backbone_cfg,
                 instance_decoder_cfg, 
                 burn_decoder_cfg):
        super().__init__()
        self.backbone_name = backbone_cfg['name'].lower()
        self.backbone_arch = backbone_cfg.get('arch', None)
        self.backbone_depth = backbone_cfg.get('depth', None)
        self.backbone_chs = backbone_cfg['out_channels']

        self.backbone = select_backbone(
            self.backbone_name, self.backbone_arch, self.backbone_depth)
        self.backbone.use_pretrained(log=False)  

        self.fpn = FPN(in_channels=self.backbone_chs, channels=256)

        self.instance_decoder_name = instance_decoder_cfg['name']
        self.burn_decoder_name = burn_decoder_cfg['name']

        self.part_instance_head = select_instance_decoder(
                                                    name=instance_decoder_cfg['name'], 
                                                    num_classes=instance_decoder_cfg['num_classes'])
        
        self.burn_decode_head = select_burn_decoder(name=burn_decoder_cfg['name'], 
                                                    num_classes=burn_decoder_cfg['num_classes'], 
                                                    class_weights=burn_decoder_cfg['class_weights'], 
                                                    in_channels=self.backbone_chs)

        pixel_mean = [123.675, 116.28, 103.53]
        pixel_std = [58.395, 57.12, 57.375]
        self.register_buffer("pixel_mean", torch.tensor(
            pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(
            pixel_std).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    @torch.jit.unused
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = self._preprocess_images(batched_inputs)
        # ground truth
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
        gt_sem_seg = ImageList.from_tensors(gt_sem_seg, size_divisibility=32)

        backbone_features = self.backbone(images.tensor)
        fpn_features = self.fpn(list(backbone_features.values()))

        losses = {}
        losses.update(self.part_instance_head(
            fpn_features, images, gt_instances))
        if "AG" in self.burn_decoder_name:
            # 人体部位二值掩码
            human_mask = self._get_human_mask_train(gt_instances, padded_size=images.tensor.shape[-2:])
            fpn_features = self._mask_select(fpn_features, human_mask)
            losses.update(self.burn_decode_head(
                list(backbone_features.values()) + list(fpn_features.values()), gt_sem_seg.tensor))
        else:
            losses.update(self.burn_decode_head(
                list(backbone_features.values()), gt_sem_seg.tensor))
        return losses


    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = self._preprocess_images(batched_inputs)
        backbone_features = self.backbone(images.tensor)
        fpn_features = self.fpn(list(backbone_features.values()))

        instances_preds = self.part_instance_head.inference(
            fpn_features, images)
        if "AG" in self.burn_decoder_name:
            # 人体部位二值掩码
            human_mask = self._get_human_mask_test(instances_preds, padded_size=images.tensor.shape[-2:])
            fpn_features = self._mask_select(fpn_features, human_mask)
            seg_logits = self.burn_decode_head(list(backbone_features.values()) + list(fpn_features.values()))
        else:
            seg_logits = self.burn_decode_head(list(backbone_features.values()))
        results = self._postprocess(
            instances_preds, seg_logits, batched_inputs, images.image_sizes)
        return results
    
    @torch.jit.unused
    def backbone_forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = self._preprocess_images(batched_inputs)
        backbone_features = self.backbone(images.tensor)
        return backbone_features

    def _preprocess_images(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        move to target device -> normalized -> padded and batched
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, size_divisibility=32)
        return images


    @staticmethod
    def _postprocess(instances, seg_logits, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances and seg logits to the original size.
        image_sizes: image sizes before padding
        input_per_image(Dict): contains the original sizes (beform transform) of the images
        """
        processed_results = []
        for instances_per_image, logits_per_image, input_per_image, image_size in zip(
            instances, seg_logits, batched_inputs, image_sizes
        ):
            height, width = input_per_image.get('image_size', input_per_image.get('image').shape[-2:])
            # resize bodding box to target size and paste forground binary mask prediction
            processed_ist = detector_postprocess(
                instances_per_image, height, width, mask_threshold=0.5)
            # crop out the size before padding and resize to original size(before transform)
            processed_sem = sem_seg_postprocess(
                logits_per_image, image_size, height, width)
            processed_results.append(
                {"instances": processed_ist, 'sem_seg': processed_sem})
        return processed_results


    def _get_human_mask_train(self, results: List[Instances], padded_size: tuple):
        paded_shape = [len(results)] + [1] +list(padded_size)
        human_masks = torch.zeros(paded_shape, dtype=torch.float, device=self.device)
        for i, result in enumerate(results):
            # 训练时 用真值 获取部位掩码
            masks = result.get('gt_masks').tensor
            # 二值化
            human_mask = torch.any(masks, dim=0,keepdim=True) # 二值化
            human_masks[i, ..., : masks.shape[-2], : masks.shape[-1]].copy_(human_mask)
        return human_masks
    
    def _get_human_mask_test(self, results: List[Instances], padded_size: tuple):
        paded_shape = [len(results)] + [1] + list(padded_size)
        human_masks = torch.zeros(paded_shape, dtype=torch.float, device=self.device)
        for i, result in enumerate(results):
            # 预测时 用部位分割结果 获取部位掩码
            height, width= result.image_size
            result = detector_postprocess(result, height, width, mask_threshold=0.5)
            masks = result.get('pred_masks')
            # 二值化
            human_mask = torch.any(masks, dim=0, keepdim=True) 
            human_masks[i, ..., : masks.shape[-2], : masks.shape[-1]].copy_(human_mask)
        return human_masks

    @staticmethod
    def _mask_select(features:Dict[str, torch.Tensor], mask:torch.Tensor):
        masked_features = {}
        mask = mask.float()
        for name, feature in features.items():
            feature_mask = F.interpolate(mask, size=(feature.shape[2:]), mode='nearest')
            masked_feature = torch.mul(feature, feature_mask)
            masked_features.update({f"masked_{name}" : masked_feature})
        return masked_features