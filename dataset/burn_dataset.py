import os
import numpy as np
import albumentations as A
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
from typing import List
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, RandomSampler
from detectron2.structures import Boxes, Instances, BitMasks


def get_data_loader(split: int|str = 0,
                    batch_size: int = 4, 
                    longest_max_size: List[int] = [384, 448, 512],
                    prefix: List[int] = ['train', 'test']):
    
    cur_dir = os.path.dirname(__file__)
    coco_json_file = f'{cur_dir}/burns/part_coco/annotations.json'
    image_dir = f'{cur_dir}/burns/image'
    sem_seg_gt_dir = f'{cur_dir}/burns/depth/label'
    train_split_file = f'{cur_dir}/burns/split/{split}/{prefix[0]}.txt'
    test_split_file = train_split_file.replace('train', f'{prefix[1]}')
    # 获取标注
    train_data_dicts = load_dataset_dicts(
        coco_json_file, image_dir, sem_seg_gt_dir, train_split_file)
    test_data_dicts = load_dataset_dicts(
        coco_json_file, image_dir, sem_seg_gt_dir, test_split_file)
    # 定义数据增强
    train_trans = A.Compose([
        A.BBoxSafeRandomCrop(erosion_rate=0.2, p=0.5), # 随即裁剪
        A.RandomBrightnessContrast(),  # 随机亮度、对比度变换
        A.LongestMaxSize(longest_max_size), # 随机缩放(固定长边)
        A.VerticalFlip(p=0.5), # 随机垂直镜像
        A.HorizontalFlip(p=0.5)], # 随机水平镜像
        bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
    
    test_trans = A.Compose([
        A.LongestMaxSize(448)],
        bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
    # 数据集
    train_ds = BurnDataset(train_data_dicts, train_trans)
    test_ds = BurnDataset(test_data_dicts, test_trans)
    # 数据加载
    train_dl = DataLoader(train_ds, 
                          batch_size, 
                          num_workers=2, 
                          sampler=RandomSampler(train_ds), 
                          collate_fn=lambda x: x,
                          drop_last=True)
    test_dl = DataLoader(test_ds, 
                         batch_size, 
                         num_workers=2, 
                         sampler=RandomSampler(test_ds), 
                         collate_fn=lambda x: x,
                         drop_last=True)
    return train_dl, test_dl


class BurnDataset(Dataset):
    def __init__(self, dataset_dicts, transform):
        super().__init__()
        self.dataset_dicts = dataset_dicts
        self.transform = transform

    def __len__(self):
        return len(self.dataset_dicts)

    def __getitem__(self, idx):
        data_dict = self.dataset_dicts[idx]
        img = np.array(Image.open(data_dict['file_name']))  # rgb
        sem_gt = np.asarray(Image.open(data_dict['sem_seg_gt']))
        # binary burn segmentation
        # sem_gt = np.array(sem_gt > 0, dtype=np.uint8)
        height, width = data_dict['height'], data_dict['width']
        img_size = (height, width)
        anns = data_dict['annotations']
        cat_ids = [ann['category_id'] for ann in anns]
        bboxes = [ann['bbox'] for ann in anns]
        mask_polygons = [ann['segmentation'][0] for ann in anns]
        num_instances = len(cat_ids)

        mask = np.zeros(img_size + (num_instances+1,))  # HxWxN+1
        mask[..., -1] = sem_gt  # 语义分割标签
        for i, mask_polygon in enumerate(mask_polygons):
            canvas = Image.new(mode='L', size=(img_size[1], img_size[0]))
            draw = ImageDraw.Draw(canvas)
            draw.polygon(mask_polygon, fill=1)
            mask[..., i] = np.asarray(canvas)

        transformed = self.transform(
            image=img, bboxes=bboxes, category_ids=cat_ids, mask=mask)
        tfm_image = transformed['image']
        tfm_gt_sem = transformed['mask'][..., -1]
        tfm_ist_mask = transformed['mask'][..., :-1]
        tfm_boxes = transformed['bboxes']
        tfm_classes = transformed['category_ids']

        tfm_boxes = torch.tensor(tfm_boxes)  # 转为张量
        tfm_boxes = torch.hstack(
            [tfm_boxes[:, :2], tfm_boxes[:, :2] + tfm_boxes[:, 2:]])  # xywh -> xyxy 格式转换

        # HxWx3 uint8 array 转为 3xHxW float32 tensor
        tfm_image = torch.tensor(
            tfm_image.transpose(2, 0, 1), dtype=torch.float32)
        tfm_classes = torch.tensor(
            tfm_classes, dtype=torch.int64)  # list 转为 int张量
        tfm_ist_mask = torch.tensor(tfm_ist_mask.transpose(
            2, 0, 1), dtype=torch.int64)  # HxWxN uint8 array 转为 NxHxW int64 张量
        # HxW uint8 array 转为 HxW int64 张量
        tfm_gt_sem = torch.tensor(tfm_gt_sem, dtype=torch.int64)

        instances = Instances(img_size)
        instances.gt_classes = tfm_classes
        tfm_boxes = Boxes(tfm_boxes)
        instances.gt_boxes = tfm_boxes
        instances.gt_masks = BitMasks(tfm_ist_mask)

        ret = {}
        ret['file_name'] = data_dict['file_name']
        ret['image_id'] = data_dict['image_id']
        ret['sem_gt_file'] = data_dict['sem_seg_gt']
        ret['image'] = tfm_image
        ret['image_size'] = img_size
        ret['sem_seg'] = tfm_gt_sem
        ret['instances'] = instances
        # ret['aux_gt_sem'] = instance_mask2semantic(masks=tfm_ist_mask, labels=tfm_classes)
        return ret


def instance_mask2semantic(masks: Tensor, labels: Tensor):
    assert masks.ndim == 3, f"{masks}'s shape must be NHW, but get {masks.shape}"
    sem_mask = torch.zeros(masks.shape[-2:], dtype=torch.int64) # HW
    for mask, label in zip(masks, labels):
        mask = mask > 0
        sem_mask[mask] = label + 1 # 语义mask 
    return sem_mask


def load_dataset_dicts(json_file, image_dir, sem_gt_dir, image_split_file=None):
    """
    Args:
    json_file: coco format instance annotations file
    image_dir: image save directory
    sem_gt_dir: semantic ground truth (grayscale) save directory, value in [0,C)
    image_split_file: Optional, json or txt file, for dataset split 
    eg: contents in image_split_file
    1.jpg
    2.jpg
    3.jpg

    Note: this function does not read image or ground truth
    """
    coco_api = COCO(json_file)
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    # for img in imgs
    # keys:['license', 'url', 'file_name', 'height', 'width', 'date_captured', 'id']
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    # for anns_per_img in anns for ann in anns_per_img
    # keys:['id', 'image_id', 'category_id', 'segmentation', 'area', 'bbox', 'iscrowd']
    cat_ids = sorted(coco_api.getCatIds())
    categories = coco_api.loadCats(cat_ids)
    thing_classes = [c["name"]
                     for c in sorted(categories, key=lambda x: x["id"])]

    imgs_anns = list(zip(imgs, anns))
    dataset_dicts = []
    ann_keys = ["category_id", "bbox", "is_crowd", "segmentation"]

    if image_split_file is not None:
        with open(image_split_file, 'r') as f:
            img_list = [item.replace('\n', '') for item in f.readlines()]
    else:
        img_list = []
    for (img_dict, anno_dict_list) in imgs_anns:
        img_name = img_dict["file_name"]
        if len(img_list) > 0 and img_name not in img_list:  # 数据集划分
            continue
        record = {}
        image_id = record["image_id"] = img_dict["id"]
        record["file_name"] = os.path.join(image_dir, img_name)
        record["sem_seg_gt"] = os.path.join(
            sem_gt_dir, img_name.replace('jpg', 'png'))
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            obj = {key: anno[key] for key in ann_keys if key in anno}
            obj["bbox_mode"] = "XYWH_ABS"
            objs.append(obj)
        record["annotations"] = objs
        record["thing_classes"] = thing_classes
        dataset_dicts.append(record)
    # print(f"Done loading {len(dataset_dicts)} samples from '{json_file}'")
    return dataset_dicts
