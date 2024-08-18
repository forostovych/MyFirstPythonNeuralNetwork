import numpy as np
import torch

def cutout(image, cutout_size=16):
    h, w = image.shape[1:]
    y = np.random.randint(h)
    x = np.random.randint(w)
    y1 = np.clip(y - cutout_size // 2, 0, h)
    y2 = np.clip(y + cutout_size // 2, 0, h)
    x1 = np.clip(x - cutout_size // 2, 0, w)
    x2 = np.clip(x + cutout_size // 2, 0, w)
    image[:, y1:y2, x1:x2] = 0
    return image

def mixup(data, target, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = data.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_data = lam * data + (1 - lam) * data[index, :]
    target_a, target_b = target, target[index]
    return mixed_data, target_a, target_b, lam

def cutmix(data, target, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(data.size()[0]).cuda()
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
    return data, target_a, target_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
