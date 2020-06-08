README

Reading file:
<< produceDataPath**.py >>
- saves a text-file with data paths, in subject-level

###################
Training file:
<< structureTrain.py >>
-------------------
Trains to segment White and Gray Matter structures from 2 MRI images.
img1 shows GM-contrast and img2 shows WM contrast.
Shuffles the subjects in text file, saves 10% for Testing,
from 90%, 20% of subjects are saved for Validation, 80 % for Training.
Training & Validation & Test data is flattened in slice-level, made as dataset using << DatasetFull.py >>
During training, losses are calculated using << losses.py >>.
Cross Entropy loss, Dice similarity loss and Hausdorff distance is used to calculate the average Score of the epoch. 
During Validation, model giving best score, meaning high Dice similarity and low Hausdorff distance, is saved as *.pt file.

Output of a model is passed through Softmax function and then Argmax to save it with classes [0, 1, 2]

##################
Evaluation file:
<< evalStructure.py >>
------------------
Evaluates the saved *.pt file using the saved Test data. 
Also used to create predictions for un-labeled input data.
Saves the predictions using Softmax as *.png and additionally using Argmax to save *.nii.gz files.
