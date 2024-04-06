import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def calc_cm(gts, preds, num_classes):
    """
    gts(list[HW array]): ground truth, value->[0,N)
    preds(list[HW array]): predictions, value->[0,N)
    num_classes(int): total classes numbe, N
    """
    cm = np.zeros((num_classes, num_classes))
    labels = np.arange(num_classes)
    for i in range(len(gts)):
        cm += metrics.confusion_matrix(
            gts[i].flatten(), preds[i].flatten(), labels=labels
        )
    return cm


def draw_cm(cm, num_classes, cmap=plt.cm.YlGnBu, sav_dir='./'):
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 13  # 设置字体样式、大小
    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 新建画布
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)  # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=list(range(num_classes)), yticklabels=list(range(num_classes)))

    data_dict = dict(weight='light', style='italic')
    ax.set_xlabel('Predicted', fontdict=data_dict)
    ax.set_ylabel('Actual', fontdict=data_dict)
    ax.set_title('Confusion Matrix', fontdict=data_dict)

    # 绘制格网
    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    ax.grid(True, which='minor', linestyle='-', color='gray')
    ax.tick_params(which="minor", bottom=False, left=False)

    # 显示百分比信息
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{(cm[i, j]*100) :.1f}%",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.savefig(f'{sav_dir}/cm.png', dpi=300, bbox_inches='tight')
