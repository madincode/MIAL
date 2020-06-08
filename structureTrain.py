# Team 1
# Training loop to segment GM & WM structures from MRI images

import numpy as np
import torch as th
import torch.optim as optim
import nibabel as nib
# import matplotlib.pyplot as plt
import json
import random
import torchvision
import time
import copy
from Unet import ResNetUNet
from collections import defaultdict
import datetime as dt
import sys
import os
# import visdom as vis
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import datasetFull as ds
from losses import *
np.set_printoptions(precision=2, suppress=True)

# Saving the training starting date in case it runs over night
dateToday = dt.date.today().strftime("%Y%m%d")


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


##################################################################
# Training loop
def train_model(dataloaders,
                device,
                model,
                criterion,
                optimizer,
                scheduler,
                softmax,
                path,
                modelname,
                num_epochs=25):
    filetmp = path + '%s%s%s.png'
    best_model_wts = copy.deepcopy(model.state_dict())
    # best_loss = 1e10  # to save best loss-based performance
    best_score = 1e-10  # to save best score-based performance

    for epoch in range(num_epochs):
        print('-' * 20)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            metrics = defaultdict(float)
            epoch_samples = 0
            sumLoss = 0
            sumCe = 0
            dicesGM = []
            dicesWM = []
            HGMs = []
            HWMs = []

            for item in dataloaders[phase]:
                inputs = item['inputs'].to(device)
                labels = item['labels'].to(device)
                name = item['name'][0]
                # print('Label shape:',labels.shape)  # [B, 1, H, W]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with th.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)  # [B, C, H, W]

                    # outputs = outputs.squeeze(0)
                    # print('Label shape:', labels.shape)
                    # print('Output shape:', outputs.shape)

                    # labels = labels.squeeze(0)
                    loss, ce, dice_gm, dice_wm = calc_loss(outputs, labels, criterion)
                    sumLoss += loss.item()
                    sumCe += ce.item()
                    dicesGM.append(dice_gm.item())
                    dicesWM.append(dice_wm.item())

                    print('Loss: %.5f ' % loss, end='')
                    print('CE: %.5f ' % ce)
                    print('DiceGM: %.5f ' % dice_gm, end='')
                    print('DiceWM: %.5f ' % dice_wm)

                    # To save the output as .png
                    outputs = softmax(outputs)
                    outputs = outputs.squeeze(0)  # [C, H, W]

                    # To save the output as in label shape [1, H, W]
                    correct = th.argmax(outputs, dim=0)
                    # print('corrected shape:', correct.shape)

                    ########################
                    # Calculate Hausdorff Distance
                    labels = labels.squeeze(0)  # [1, H, W]
                    hausdorffGM = symHausdorff(correct == 2, labels == 2)
                    hausdorffWM = symHausdorff(correct == 1, labels == 1)
                    HGMs.append(hausdorffGM)
                    HWMs.append(hausdorffWM)
                    print('Hausdorff GM: %.5f, Hausdorff WM: %.5f'
                          % (hausdorffGM, hausdorffWM))
                    print('-----------------------')

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # outputs.backward()
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)  # number of samples trained

            print_metrics(metrics, epoch_samples, phase)
            # epoch_loss = sumLoss / epoch_samples  # to save best loss-based performance
            diceMeanGM, meanHGM, scoreGM = calc_score(dicesGM, HGMs)
            diceMeanWM, meanHWM, scoreWM = calc_score(dicesWM, HWMs)
            epoch_score = .5 * (scoreGM + scoreWM)  # to save best score-based performance
            print('Score: ', epoch_score)

            print('Average loss: %.5f, Average Ce: %.5f, Average Dice: %.5f'
                  % (sumLoss/epoch_samples, sumCe/epoch_samples, (diceMeanGM+diceMeanWM)/2))
            print('Average HGM: %.5f, Average HWM: %.5f'
                  % (meanHGM, meanHWM))

            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
            # deep copy the model
            # if phase == 'val' and epoch_loss < best_loss:  # to save best loss-based performance
            if phase == 'val' and epoch_score > best_score:  # to save best score-based performance
                print("saving best model")  # save a hard copy of the model!
                # best_loss = epoch_loss  # to save best loss-based performance
                best_score = epoch_score  # to save best score-based performance
                best_model_wts = copy.deepcopy(model.state_dict())
                th.save(model.state_dict(), path + modelname
                        + str(epoch+1) + 'best' + '.pt')

        # Save the last image as .png (optional)
        torchvision.utils.save_image(
            outputs,
            filetmp % (modelname,
                       str(epoch+1),
                       name),
            normalize=True)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # To save each epoch model independently from best model (optinal)
        th.save(model.state_dict(), path + modelname
                + str(epoch+1) + '.pt')

    print('Best val loss: {:4f}'.format(best_score))  # or best_loss

    ##################################################################
    # load best model weights and save them
    model.load_state_dict(best_model_wts)
    return model


def main():
    # Give model a name
    modelname = 'Training'

    ##################################################################
    # Step 1. Load Dataset (saved as a list in a text file)
    train_data_full = json.load(open('data_mialab/data_subjects.txt'))
    random.shuffle(train_data_full)  # shuffle subject-wise
    print('Num. of subjects ...', len(train_data_full))

    # save 90% for training and 10% for testing
    train_data, test_data = ds.splitData(train_data_full, proportion=0.9)

    # print('Num. of subjects in train data ...', len(train_data))

    # separating patients and healthy subjects
    patients = [sub for sub in train_data if sub['is_patient']]
    healthy = [sub for sub in train_data if not sub['is_patient']]

    # shuffle the lists
    random.shuffle(patients)
    random.shuffle(healthy)

    # extract 80% for training and 20% for validation from each group
    training_patients, validation_patients = ds.splitData(patients)
    training_healthy, validation_healthy = ds.splitData(healthy)

    # putting together shuffles patients and HC
    training_data = training_patients + training_healthy
    validation_data = validation_patients + validation_healthy

    # flattened training data into slice level
    training_slices = ds.flattenSlices(training_data)
    validation_slices = ds.flattenSlices(validation_data)
    test_slices = ds.flattenSlices(test_data)

    # create training dataset
    augrate = 2  # augrate - 1 = nr. of augmentations of the original image
    training_dataset = ds.Dataset(training_slices, augrate=augrate)
    validation_dataset = ds.Dataset(validation_slices, train=False)

    #  Print the length of data
    print('Num. of subjects in training data ...', len(training_data))
    print('Num. of patients slices in training data...',
          len([s for s in training_slices if s['is_patient']]))
    print('Num. of healthy slices in training data...',
          len([s for s in training_slices if not s['is_patient']]))
    print('Num. of slices in training dataset (x%.0f with augmentation) ...'
          % augrate, len(training_dataset))
    print('Num. of subjects in validation data ...', len(validation_data))
    print('Num. of slices validation dataset ...', len(validation_dataset))
    print('Num. of subjects in test data...', len(test_data))
    print('Num. of slices in test data...', len(test_slices))

    ##################################################################
    # Step 2. Instantiate Model Class
    device = th.device('cuda: 0' if th.cuda.is_available() else 'cpu')
    print('Device:', device)
    batch_size = 1  # currently, loss function only works with size=1
    print('Batch size:', batch_size)
    num_class = 3  # background, WM, GM
    epochs = 50
    lr_rate = 1e-4
    model = ResNetUNet(num_class).to(device)
    modelname = dateToday + modelname + str(epochs) + 'e'
    print('Model name:', modelname)
    path = 'CodeStructure/Train_outputs/'+modelname+'/'

    # Create a path to save outputs
    if not os.path.exists(path):
        os.makedirs(path)

    # saving the test data
    json.dump(test_slices, open(path + modelname
                                + 'test_data.txt', "w"))
    # json.dump(test_slices, open('madina/mial_code/'
    #                             + dateToday + modelname
    #                             + 'test_data.txt', "w"))

    ##################################################################
    # Step 3. Make Dataset Iterable
    dataloaders = {
        'train': DataLoader(dataset=training_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0),
        'val': DataLoader(dataset=validation_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
    }

    ##################################################################
    # 4. Choose optimizer & loss function
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad,
                                     model.parameters()),
                              lr=lr_rate)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                           step_size=30,
                                           gamma=0.1)

    # weight = th.Tensor([0.05, 0.35, 0.60]).to(device)  # to assign weight to class losses
    weight = None
    criterion = th.nn.CrossEntropyLoss(weight=weight)
    softmax = th.nn.Softmax(dim=1)

    ##################################################################
    # 5. Use the model to train
    model = train_model(dataloaders,
                        device,
                        model,
                        criterion,
                        optimizer_ft,
                        exp_lr_scheduler,
                        softmax,
                        path,
                        modelname,
                        num_epochs=epochs)


if __name__ == '__main__':
    main()
