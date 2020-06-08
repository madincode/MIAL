# Team 1
# Modified Dataset from Pytorch

import numpy as np
import torch as th
import nibabel as nib
import json
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
np.set_printoptions(precision=2, suppress=True)


class Dataset(Dataset):
    def __init__(self, slices, augrate=4, train=True):
        self.slices = slices
        self.augRate = augrate
        self.train = train
        self.params = []
        for sli in range(len(self.slices)):
            numAug = self.augRate
            if not self.train:
                numAug = 1  # only original images during evaluation

            for aug in range(numAug):
                self.params.append((sli, aug))

    def __getitem__(self, index):
        # for indexing the dataset
        s, aug = self.params[index]

        sli = self.slices[s]  # slice has 2 inputs & 1 structure label

        im1 = (loadNiftiToTensor(sli['img1']))
        im2 = (loadNiftiToTensor(sli['img2']))
        x = th.stack([im1, im2], dim=0)  # stacking together the 2 inputs

        subject_pos = sli['slice'].index('subject')
        path = sli['slice'][subject_pos:]
        name = path.replace('/', '_')  # subjextXX_(scanXX)_sliceXX

        if self.train:
            y = loadNiftiToTensor(sli['structure'])
            # x, y = rotate(x, y, aug)
            x, y = augment(x, y, aug)
            # return {'inputs': x, 'labels': y, 'name': name}
        else:
            if 'structure' in sli:  # add empty label if structure doesn't exist
                y = loadNiftiToTensor(sli['structure'])
            else:
                y = getEmpty(sli, im1.shape)
        return {'inputs': x, 'labels': y, 'name': name}

    def __len__(self):
        return len(self.params)


def rotate(inputs, labels, aug):
    # make a better rotation!
    # print('input shape:', inputs.shape)
    # print('label shape:', labels.shape)
    if aug == 0:
        return (inputs, labels)
    elif aug == 4:
        inputs = addNoise(inputs)
        return (inputs, labels)
    elif aug == 1:
        inputs = th.rot90(inputs, 1, [1, 2])
        labels = th.rot90(labels, 1, [0, 1])
        # return (inputs, labels)
    elif aug == 2:
        inputs = th.rot90(inputs, 2, [1, 2])
        labels = th.rot90(labels, 2, [0, 1])
        # return (inputs, labels)
    elif aug == 3:
        inputs = th.rot90(inputs, 3, [1, 2])
        labels = th.rot90(labels, 3, [0, 1])
    # if random.random() > .5:
    # inputs = addNoise(inputs, nf=0)  #unindent when rotate is on
    return (inputs, labels)


# can put various types of augmentations together
def augment(inputs, labels, aug):
    if aug == 0:
        return (inputs, labels)
    # elif aug == 1:
    #     rot = random.randint(0, 3)
    #     inputs = th.rot90(inputs, rot, [1, 2])
    #     labels = th.rot90(labels, rot, [0, 1])
    #     inputs = addNoise(inputs)
        # return (inputs, labels)
    else:
        # rot = random.randint(1, 3)
        # inputs = th.rot90(inputs, rot, [1, 2])
        # labels = th.rot90(labels, rot, [0, 1])
        inputs = addNoise(inputs, nf1=0.02, nf2=0.15)
        # return (inputs, labels)
    # elif aug == 3:
    #     inputs = th.rot90(inputs, 3, [1, 2])
    #     labels = th.rot90(labels, 3, [0, 1])
    # if random.random() > .5:
    # inputs = addNoise(inputs, nf=0)  #unindent when rotate is on
    return (inputs, labels)


def addNoise(image, nf1=0.05, nf2=0.15):
    grain_size = random.randint(10, 25)
    noise_factor = random.uniform(nf1, nf2)

    upsample = th.nn.Upsample(size=image.shape[-2:], mode='bicubic')
    noise = upsample((th.rand(grain_size, grain_size)*1.4-1).view(1, 1, grain_size, grain_size))[0, 0]
    noise = th.stack([noise, noise], dim=0)
    return image + noise*noise_factor


def getEmpty(sli, shape):
    return th.zeros(shape, dtype=th.float64)


def loadNiftiToTensor(path):
    niftiFile = nib.load(path)
    # should it save the affine & header info of the image?
    # which is needed for saving nifti images later
    # affine = niftiFile.affine --> create an empty affine matrix
    img = th.Tensor(niftiFile.get_fdata())
    return img


def splitData(data, proportion=0.8):
    n = len(data)
    split_point = int(round(n * proportion))
    return data[:split_point], data[split_point:]


def flattenSlices(data):
    slices = []
    for s in data:
        slices += s['slices']
    return slices


# To test the dataset:
def main():
    train_data_full = json.load(open('data_mialab/data_subjects.txt'))
    # train_data_full = json.load(open('data_mialab/data_short.txt'))
    random.shuffle(train_data_full)
    print('Num. of subjects ...', len(train_data_full))

    patients = [sub for sub in train_data_full if sub['is_patient']]
    healthy = [sub for sub in train_data_full if not sub['is_patient']]

    training_slices = flattenSlices(train_data_full)
    training_dataset = Dataset(training_slices, augrate=2, train=True)
    print('Num. of patients in training data...', len(patients))
    print('Num. of healthy in training data...', len(healthy))
    print('Num. of patients slices in training data...',
          len([s for s in training_slices if s['is_patient']]))
    print('Num. of healthy slices in training data...',
          len([s for s in training_slices if not s['is_patient']]))
    print('Num. of slices in training dataset (x4 with augmentation) ...',
          len(training_dataset))

    for item in training_dataset:
        inputs = item['inputs']
        labels = item['labels']
        plt.subplot(131)
        plt.imshow(inputs[0], cmap='gray')
        plt.ylabel('segemented images')
        plt.xlabel('width (pixels)')
        plt.subplot(132)
        plt.imshow(inputs[1], cmap='gray')
        plt.subplot(133)
        # plt.imshow(labels.permute(1, 2, 0), cmap='gray')
        plt.imshow(labels, cmap='gray')
        # vis.matplot(plt, win=image3)
        plt.pause(0.001)
        # break
    plt.show()
        # vis.close(win=textwindow)
        # vis.close(win=image3)


if __name__ == '__main__':
    main()
