import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import _LRScheduler

import torch
import torch.nn.functional as F
from piq import MultiScaleSSIMLoss


def show_img(img, title=None, figsize=(15, 15)):
  #img = (img+1)/2
  plt.figure(figsize=figsize)
  plt.imshow(img)
  if title:
    plt.title(title)
  plt.show()

class ShiftMSSSIM(torch.nn.Module):
  """Shifted SSIM Loss """
  def __init__(self):
    super(ShiftMSSSIM, self).__init__()
    self.ssim = MultiScaleSSIMLoss(data_range=1.)

  def forward(self, est, gt):
    # shift images back into range (0, 1)
    est = est * 0.5 + 0.5
    gt = gt *0.5 + 0.5
    return self.ssim(est, gt)

# Rain Robust Loss
# Code modified from: https://github.com/sthalles/SimCLR/blob/master/simclr.py

class RainRobustLoss(torch.nn.Module):
  """Rain Robust Loss"""
  def __init__(self, batch_size, n_views, device, temperature=0.07):
    super(RainRobustLoss, self).__init__()
    self.batch_size = batch_size
    self.n_views = n_views
    self.temperature = temperature
    self.device = device
    self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

  def forward(self, features):
    logits, labels = self.info_nce_loss(features)
    return self.criterion(logits, labels)

  def info_nce_loss(self, features):
    labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(self.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

    logits = logits / self.temperature
    return logits, labels

"""
Linear warmup from: 
https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
"""

class GradualWarmupScheduler(_LRScheduler):
  """ Gradually warm-up(increasing) learning rate in optimizer.
  Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
  Args:
      optimizer (Optimizer): Wrapped optimizer.
      multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
      total_epoch: target learning rate is reached at total_epoch, gradually
      after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
  """

  def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
    self.multiplier = multiplier
    if self.multiplier < 1.:
      raise ValueError('multiplier should be greater thant or equal to 1.')
    self.total_epoch = total_epoch
    self.after_scheduler = after_scheduler
    self.finished = False
    super(GradualWarmupScheduler, self).__init__(optimizer)

  def get_lr(self):
    if self.last_epoch > self.total_epoch:
      if self.after_scheduler:
        if not self.finished:
          self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
          self.finished = True
        return self.after_scheduler.get_last_lr()
      return [base_lr * self.multiplier for base_lr in self.base_lrs]

    if self.multiplier == 1.0:
      return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
    else:
      return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

  def step_ReduceLROnPlateau(self, metrics, epoch=None):
    if epoch is None:
      epoch = self.last_epoch + 1
    self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
    if self.last_epoch <= self.total_epoch:
      warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
      for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
        param_group['lr'] = lr
    else:
      if epoch is None:
        self.after_scheduler.step(metrics, None)
      else:
        self.after_scheduler.step(metrics, epoch - self.total_epoch)

  def step(self, epoch=None, metrics=None):
    if type(self.after_scheduler) != ReduceLROnPlateau:
      if self.finished and self.after_scheduler:
        if epoch is None:
          self.after_scheduler.step(None)
        else:
          self.after_scheduler.step(epoch - self.total_epoch)
        self._last_lr = self.after_scheduler.get_last_lr()
      else:
        return super(GradualWarmupScheduler, self).step(epoch)
    else:
      self.step_ReduceLROnPlateau(metrics, epoch)