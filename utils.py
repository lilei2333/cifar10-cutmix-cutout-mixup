import os
import time
import shutil

import numpy as np
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def update(self, sum, n):
        self.sum += sum
        self.count += n
        self.avg = 0 if self.count == 0 else self.sum / self.count * 100.0


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    if os.path.exists(path):
        if remove:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k)
    num_map = generate_map()
    correct_map = generate_map()
    n_correct = correct[:k].view(-1)
    for i in range(batch_size):
        t = target[i].item()
        num_map[t] += 1
        if n_correct[i].item():
            correct_map[t] += 1
    return res, num_map, correct_map


def generate_map():
    return {i: 0 for i in range(10)}


def generate_fig(t):
    fig = plt.figure(1)
    ax1 = plt.subplot(111)
    d = []
    for i in range(10):
        d.append(t[i].avg)
    data = np.array(d)
    width = 0.5
    x_bar = np.arange(10)
    rect = ax1.bar(x=x_bar, height=data, width=width, color="lightblue")
    for rec in rect:
        x = rec.get_x()
        height = rec.get_height()
        ax1.text(x + 0.1, 1.02 * height, str(np.round(height, 2)))
    ax1.set_xticks(x_bar)
    ax1.set_xticklabels(("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"),
                        rotation=45)
    ax1.set_ylabel("acc (%)")
    ax1.set_title("Accurate rate for each class", y=1.07)
    ax1.grid(True)
    ax1.set_ylim(0, 100)
    plt.subplots_adjust(bottom=0.15)
    # plt.show()
    return fig


# for data augmentation: mixup cutout cutmix

def mixup_data(x, y, alpha=1.0, device=torch.device("cuda")):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    index = index.to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1. - lam) * criterion(pred, y_b)


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            # (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def cutmix_data(x, y, alpha=1.0, device=torch.device("cuda")):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    rand_index = torch.randperm(x.size()[0]).to(device)
    y_a = y
    y_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = _rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1. - lam) * criterion(pred, y_b)


def _rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
