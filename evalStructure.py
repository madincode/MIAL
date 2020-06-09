# Team 1
# Evaluate the trained model and save the output images
# 10% of train subjects

import torch as th
from torch.utils.data import DataLoader
import torchvision
from Unet import ResNetUNet
import json
import nibabel as nib
import sys
import os
import datasetFull as ds
from losses import *


def main():
    online = 'online' in sys.argv[1:]
    print('Online evaluation?', online)
    if online:
        save = 'Online/'
        test_data_path = 'data_mialab/data_online.txt'  # for the online evaluation dataset
    else:
        save = ''

    #####################################
    # choose main model here
    model = '20200527ModelCeDiceN50e'
    best = 11  # choose the epoch from best model saved
    modelname = model + str(best)
    #####################################

    model_path = 'MIAL/Train_outputs/'+model+'/'+modelname+'.pt'

    test_data_path = 'MIAL/Train_outputs/'+model+'/'+model+'test_data.txt'
    if online:
        test_data_path = 'data_mialab/data_online.txt'  # for the online evaluation dataset

    # Create a path to save outputs
    goal_path = 'MIAL/Test_outputs/' + modelname + '/' + save
    if not os.path.exists(goal_path):
        os.makedirs(goal_path)
    print('Results are saved in: ', goal_path)

    # Load dataset for test (already flattened slice-wise)
    test_data = json.load(open(test_data_path))

    print('Evaluating model: ', modelname)
    print('Test data: ', test_data_path)

    # create test dataset
    test_dataset = ds.Dataset(test_data, train=False)
    print('length of test dataset', len(test_dataset))

    # Step 2. Instantiate Model Class and set to evaluation mode
    device = th.device('cuda: 0' if th.cuda.is_available() else 'cpu')
    print('Device:', device)
    batch_size = 1
    print('Batch size:', batch_size)
    num_class = 3

    # load the saved model
    model = ResNetUNet(num_class).to(device)
    model.load_state_dict(th.load(model_path, map_location=device))
    model.eval()  # Set model to the evaluation mode

    # load data
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)

    criterion = th.nn.CrossEntropyLoss()
    softmax = th.nn.Softmax(dim=1)
    count = 0
    sumLoss = 0
    sumCe = 0
    dicesGM = []
    dicesWM = []
    HGMs = []
    HWMs = []

    with th.no_grad():
        for item in test_loader:
            count += 1
            inputs = item['inputs'].to(device)
            labels = item['labels'].to(device)
            name = item['name'][0]
            # print('filename:', name)
            pred = model(inputs)

            ########################
            # Calculate and print losses
            loss, ce, dice_gm, dice_wm = calc_loss(pred, labels, criterion)
            sumLoss += loss.item()
            sumCe += ce.item()
            dicesGM.append(dice_gm.item())
            dicesWM.append(dice_wm.item())
            print('Loss: %.5f ' % loss, end='')
            print('CE: %.5f ' % ce)
            print('DiceGM: %.5f ' % dice_gm, end='')
            print('DiceWM: %.5f ' % dice_wm)

            # To save the prediction as .png
            pred = softmax(pred)
            pred = pred.squeeze(0)  # [C, H, W]

            # To save the output as in label shape [1, H, W]
            correct = th.argmax(pred, dim=0)

            ########################
            # Calculate Hausdorff Distance
            labels = labels.squeeze(0)
            hausdorffGM = symHausdorff(correct == 2, labels == 2)
            hausdorffWM = symHausdorff(correct == 1, labels == 1)
            HGMs.append(hausdorffGM)
            HWMs.append(hausdorffWM)
            print('Hausdorff GM: %.5f, Hausdorff WM: %.5f'
                  % (hausdorffGM, hausdorffWM))
            print('------------------------------')

            ########################
            # Save the prediction as .png
            torchvision.utils.save_image(pred, goal_path
                                         + name + '_' + modelname + '.png',
                                         normalize=True)

            # Save the corrected prediction as nifti
            # affine = th.eye(4)
            nib_img = nib.Nifti1Image(correct.cpu().detach().numpy().astype('int16'), None)
            nib.save(nib_img, goal_path
                     + name + '_' + modelname + '.nii.gz')

    #########################
    # Calculate final score and losses
    diceMeanGM, meanHGM, scoreGM = calc_score(dicesGM, HGMs)
    diceMeanWM, meanHWM, scoreWM = calc_score(dicesWM, HWMs)
    score = .5 * (scoreGM + scoreWM)
    print('Score: ', score)
    print('Average Loss: %.5f, Average Ce: %.5f, Average Dice: %.5f'
          % (sumLoss/count, sumCe/count, (diceMeanGM+diceMeanWM)/2))
    print('Average HGM: %.5f, Average HWM: %.5f'
          % (meanHGM, meanHWM))
    print('------------------------------')


if __name__ == '__main__':
    main()
