import numpy as np
import random

__all__ = ["fixed_colormap", "random_color", "random_colors"]
# fmt: off
# RGB:
_COLORS = np.array([
                [192, 64, 0],
                [64, 192, 0],
                [192, 192, 0],
                [192, 64, 128],
                [64, 192, 128],
                [0, 0, 192],
                [128, 0, 192],
                [0, 128, 192],
                [192, 0, 64],
                [192, 128, 64],
                [64, 0, 192],
                [192, 0, 192],
                [64, 128, 192],
                [0, 192, 64],
                [128, 192, 64],
                [0, 64, 192],
                [128, 64, 192],
                [0, 192, 192],
                [192, 64, 64],
                [64, 192, 64],
                [192, 192, 64],
                [64, 64, 192],
                [192, 64, 192],
                [64, 192, 192],
                [160, 0, 0],
                [160, 128, 0],
                [160, 0, 128],
                [224, 0, 0],
                [224, 128, 0],
                [224, 0, 128],
                [160, 64, 0],
                [32, 192, 0],
                [160, 192, 0],
                [160, 64, 128],
                [32, 192, 128],
                [224, 64, 0],
                [96, 192, 0],
                [224, 192, 0],
                [224, 64, 128],
                [160, 0, 64],
                [160, 128, 64],
                [32, 0, 192],
                [160, 0, 192],
                [32, 128, 192],
                [224, 0, 64],
                [224, 128, 64],
                [96, 0, 192],
                [224, 0, 192],
                [160, 64, 64],
                [32, 192, 64],
                [160, 192, 64],
                [32, 64, 192],
                [160, 64, 192],
                [32, 192, 192],
                [224, 64, 64],
                [96, 192, 64],
                [224, 192, 64],
                [96, 64, 192],
                [224, 64, 192],
                [0, 160, 0],
                [128, 160, 0],
                [0, 160, 128],
                [192, 32, 0],
                [64, 160, 0],
                [192, 160, 0],
                [192, 32, 128],
                [64, 160, 128],
                [0, 224, 0],
                [128, 224, 0],
                [0, 224, 128],
                [192, 96, 0],
                [64, 224, 0],
                [192, 224, 0],
                [64, 224, 128],
                [0, 160, 64],
                [128, 160, 64],
                [0, 32, 192],
                [128, 32, 192],
                [0, 160, 192],
                [192, 32, 64],
                [64, 160, 64],
                [192, 160, 64],
                [64, 32, 192],
                [192, 32, 192],
                [64, 160, 192],
                [0, 224, 64],
                [128, 224, 64],
                [0, 96, 192],
                [0, 224, 192],
                [192, 96, 64],
                [64, 224, 64],
                [192, 224, 64],
                [64, 96, 192],
                [64, 224, 192],
                [160, 32, 0],
                [32, 160, 0],
                [160, 160, 0],
                [160, 32, 128],
                [32, 160, 128],
                [224, 32, 0],
                [96, 160, 0],
                [224, 160, 0],
                [224, 32, 128],
                [160, 96, 0],
                [32, 224, 0],
                [160, 224, 0],
                [32, 224, 128],
                [224, 96, 0],
                [96, 224, 0],
                [224, 224, 0],
                [160, 32, 64],
                [32, 160, 64],
                [160, 160, 64],
                [32, 32, 192],
                [160, 32, 192],
                [32, 160, 192],
                [224, 32, 64],
                [96, 160, 64],
                [224, 160, 64],
                [96, 32, 192],
                [224, 32, 192],
                [160, 96, 64],
                [32, 224, 64],
                [160, 224, 64],
                [32, 96, 192],
                [32, 224, 192],
                [224, 96, 64],
                [96, 224, 64],
                [224, 224, 64],
                [0, 0, 160],
                [128, 0, 160],
                [0, 128, 160],
                [192, 0, 32],
                [192, 128, 32],
                [64, 0, 160],
                [192, 0, 160],
                [64, 128, 160],
                [0, 192, 32],
                [128, 192, 32],
                [0, 64, 160],
                [128, 64, 160],
                [0, 192, 160],
                [192, 64, 32],
                [64, 192, 32],
                [192, 192, 32],
                [64, 64, 160],
                [192, 64, 160],
                [64, 192, 160],
                [0, 0, 224],
                [128, 0, 224],
                [0, 128, 224],
                [192, 0, 96],
                [64, 0, 224],
                [192, 0, 224],
                [64, 128, 224],
                [0, 192, 96],
                [0, 64, 224],
                [128, 64, 224],
                [0, 192, 224],
                [192, 64, 96],
                [64, 192, 96],
                [64, 64, 224],
                [192, 64, 224],
                [64, 192, 224],
                [160, 0, 32],
                [160, 128, 32],
                [32, 0, 160],
                [160, 0, 160],
                [32, 128, 160],
                [224, 0, 32],
                [224, 128, 32],
                [96, 0, 160],
                [224, 0, 160],
                [160, 64, 32],
                [32, 192, 32],
                [160, 192, 32],
                [32, 64, 160],
                [160, 64, 160],
                [32, 192, 160],
                [224, 64, 32],
                [96, 192, 32],
                [224, 192, 32],
                [96, 64, 160],
                [224, 64, 160],
                [160, 0, 96],
                [32, 0, 224],
                [160, 0, 224],
                [32, 128, 224],
                [224, 0, 96],
                [96, 0, 224],
                [224, 0, 224],
                [160, 64, 96],
                [32, 192, 96],
                [32, 64, 224],
                [160, 64, 224],
                [32, 192, 224],
                [224, 64, 96],
                [96, 64, 224],
                [224, 64, 224],
                [0, 160, 32],
                [128, 160, 32],
                [0, 32, 160],
                [128, 32, 160],
                [0, 160, 160],
                [192, 32, 32],
                [64, 160, 32],
                [192, 160, 32],
                [64, 32, 160],
                [192, 32, 160],
                [64, 160, 160],
                [0, 224, 32],
                [128, 224, 32],
                [0, 96, 160],
                [0, 224, 160],
                [192, 96, 32],
                [64, 224, 32],
                [192, 224, 32],
                [64, 96, 160],
                [64, 224, 160],
                [0, 160, 96],
                [0, 32, 224],
                [128, 32, 224],
                [0, 160, 224],
                [192, 32, 96],
                [64, 160, 96],
                [64, 32, 224],
                [192, 32, 224],
                [64, 160, 224],
                [0, 224, 96],
                [0, 96, 224],
                [0, 224, 224],
                [64, 224, 96],
                [64, 96, 224],
                [64, 224, 224],
                [160, 32, 32],
                [32, 160, 32],
                [160, 160, 32],
                [32, 32, 160],
                [160, 32, 160],
                [32, 160, 160],
                [224, 32, 32],
                [96, 160, 32],
                [224, 160, 32],
                [96, 32, 160],
                [224, 32, 160],
                [160, 96, 32],
                [32, 224, 32],
                [160, 224, 32],
                [32, 96, 160],
                [32, 224, 160],
                [224, 96, 32],
                [96, 224, 32],
                [224, 224, 32],
                [160, 32, 96],
                [32, 160, 96],
                [32, 32, 224]], dtype=np.uint8)

def fixed_colormap(colors_num:int=256, normalized:bool=False, mode:str='rgb'):
    """
    Args:
        colors_num (int<=256): number of unique colors needed
        normalized (bool): whether to normalize colors
        mode (str): whether to return RGB colors or BGR colors.
    Returns:
        ndarray: a uint8 or float32 array of Nx3 colors, in range [0, 255] or [0, 1]
    """
    assert colors_num > 0 and colors_num <= len(_COLORS)
    colormap = _COLORS[:colors_num, :]
    if normalized:
        colormap = colormap.astype(np.float32)/ 255
    if mode == 'bgr':
        colormap = colormap[:,::-1]
    return colormap


def random_color(normalized:bool=False, mode:str='rgb'):
    """

    Args:
        normalized (bool): whether to normalize colors
        mode (str): whether to return RGB colors or BGR colors.
    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    color = _COLORS[idx]
    if mode == 'bgr':
        color = color[:, ::-1]
    if normalized:
        color = color.astype(np.float32)/ 255
    return color


def random_colors(N:int=12, normalized:bool=False, mode:str='rgb'):
    """
    Args:
        N (int): number of unique colors needed, N<=256
        normalized (bool): whether to normalize colors
        mode (str): whether to return RGB colors or BGR colors.
    Returns:
        ndarray: a uint8 or float32 array of N random colors, in range [0, 255] or [0, 1]
    """
    indices = random.sample(range(len(_COLORS)), N)
    colors = np.asarray([_COLORS[i] for i in indices])
    if normalized:
        colors = colors.astype(np.float32) / 255
    if mode == 'bgr':
        colors = colors[:, ::-1]
    return colors


if __name__ == "__main__":
    
    import cv2
    size = 100
    H, W = 16, 16
    canvas = np.ones([H * size, W * size, 3], dtype=np.uint8)
    # colors = random_colors(H*W)
    # colors = np.repeat(random_color()[np.newaxis,:], H*W, axis=0)
    colors = fixed_colormap(H*W)
    for h in range(H):
        for w in range(W):
            idx = h * W + w
            if idx >= len(colors):
                break
            canvas[h * size : (h + 1) * size, w * size : (w + 1) * size] = colors[idx]
    cv2.imwrite('canvas.png', canvas[:,:,::-1])