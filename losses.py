# Team 1
# Cross Entropy and Dice loss calculation
# Hausdorff distance calculation
# currently only works with batch size = 1

import numpy as np
import torch as th
# from torch.utils.data import DataLoader
from scipy.spatial.distance import directed_hausdorff
np.set_printoptions(precision=2, suppress=True)


##################################################################
# Loss calculation
def calc_loss(pred, target, criterion):
    ce = criterion(pred, target.long())  # Cross Entropy loss
    dice_gm, dice_wm = diceScore(pred, target)

    dice = (dice_wm + dice_gm) / 2
    # print('dice: ', dice.item())

    loss = ce + (1-dice)  # edit here for different weightning

    return loss, ce, dice_gm, dice_wm


def diceScore(pred, label, empty_score=1.0):
    # calculates dice score which is average dice of GM & WM
    softmax = th.nn.Softmax(dim=1)
    pred = softmax(pred)
    label = label.squeeze(0)
    pred = pred.squeeze(0)

    bool_label = th.stack([label == i for i in range(3)])
    correct_pred = th.argmax(pred, dim=0)
    bool_pred = th.stack([correct_pred == i for i in range(3)])

    pred_wm = bool_pred[1]
    label_wm = bool_label[1]

    pred_gm = bool_pred[2]
    label_gm = bool_label[2]

    dice_gm = dice_loss(pred_gm, label_gm)
    dice_wm = dice_loss(pred_wm, label_wm)

    return dice_gm, dice_wm


def dice_loss(pred, label, empty_score=0.0):
    # implements dice similarity
    # Args: prediction image & label image of the same size
    # Dice coefficient as a float on range [0,1].
    # Maximum similarity = 1
    # No similarity = 0

    intersection = th.sum(pred * label)
    # print(intersection)
    im_sum = th.sum(pred)+th.sum(label)
    # print(th.sum(pred), th.sum(label))

    if im_sum == 0:
        return th.Tensor([empty_score])

    value = 2. * intersection / im_sum

    return value


# Calculate Hausdorff distance for a class in mm
def symHausdorff(x, y):
    return max(dirHausdorff(x, y), dirHausdorff(y, x))


def dirHausdorff(x, y):
    mmperpixel = 0.134  # a pixel corresponds to 0.134 mm
    return mmperpixel * directed_hausdorff(x.cpu().numpy(), y.cpu().numpy())[0]


# Calculate the score for a class
def calc_score(dices, hausdorffs):
    dices = np.array(dices)
    diceMean = dices.mean()
    diceMed = np.median(dices)
    diceSTD = np.std(dices)
    hausMean = np.array(hausdorffs).mean()
    score = 100 * .5 * (diceMean + diceMed - diceSTD - hausMean)
    return diceMean, hausMean, score
