import torch
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt


class SemanticVisualizer:
    # R, G, B
    palette = np.array([
                        [0,0,0],
                        [0,255,0],  
                        [255,255,0], 
                        [0,0,255]], 
                        dtype=np.uint8)

    def __init__(self, image, alpha=0.5):
        self.image = image # rgb image
        self.alpha = alpha
        self.num_classes = self.palette.shape[0]
    
    def visualize_mask(self, x):
        # 判断数据类型, 目的将chw张量 转换为 hw numpy数组
        if isinstance(x, torch.Tensor):
            if x.dim() > 2: # chw -> hw -> ndarray
                x = torch.argmax(x, dim=0).detach().cpu().numpy()
            else:
                x = x.detach().cpu().numpy()
        if x.ndim > 2: # chw -> hw
            x = np.argmax(x, axis=0)
        mask = self.palette[x]
        non_zone = np.all(mask==np.zeros(3).reshape(1,1,3), axis=2)
        non_mask_zone = np.repeat(non_zone[..., None], repeats=3, axis=2)
        # 前景区mask半透明覆盖
        viz = self.image * non_mask_zone + (1- non_mask_zone) * (self.alpha * mask + (1 - self.alpha) * self.image)
        return viz.astype(np.uint8)
    
    def visualize_contour(self, x):
        """
        return: matplotlib.figure
        """
        if isinstance(x, torch.Tensor):
            if x.dim() > 2: # chw -> hw -> ndarray
                x = torch.argmax(x, dim=0).detach().cpu().numpy()
            else:
                x = x.detach().cpu().numpy()
        if x.ndim > 2: # chw -> hw
            x = np.argmax(x, axis=0)
        x = np.eye(self.num_classes)[x] # 转成one-hot, HWC
        # 初始画布 
        dpi=300 # 分辨率
        height, width = x.shape[:2]
        # 设置画布大小与image size 相同
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_axes([0,0,1,1])
        ax.set_axis_off() # 关闭坐标系
        ax.imshow(self.image) # canvas

        for i in range(1, self.num_classes): # 不考虑背景类的边界
            class_mask = x[..., i]
            if class_mask.max(): # 若max_value==0, 说明无此类
                contours = find_contours(class_mask) # skimage.measure.find_contours()
                color = (self.palette[i] / 255).tolist() # 颜色归一化, 并转换成list
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], linewidth=0.8, color=color)
        plt.close()
        return fig
    
    def visualize_mask_contour(self, x):
        """
        return: matplotlib.figure
        """
        viz = self.visualize_mask(x)
        if isinstance(x, torch.Tensor):
            if x.dim() > 2: # chw -> hw -> ndarray
                x = torch.argmax(x, dim=0).detach().cpu().numpy()
            else:
                x = x.detach().cpu().numpy()
        if x.ndim > 2: # chw -> hw
            x = np.argmax(x, axis=0)
        x = np.eye(self.num_classes)[x] # 转成one-hot, HWC
        # 初始画布 
        dpi=300 # 分辨率
        height, width = x.shape[:2]
        # 设置画布大小与image size 相同
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_axes([0,0,1,1])
        ax.set_axis_off() # 关闭坐标系
        ax.imshow(viz) # canvas
        for i in range(1, self.num_classes): # 不考虑背景类的边界
            class_mask = x[..., i]
            if class_mask.max(): # 
                contours = find_contours(class_mask) # skimage.measure.find_contours()
                color = (self.palette[i] / 255).tolist() # 颜色归一化, 并转换成list
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], linewidth=0.8, color=color)
        plt.close()
        return fig
