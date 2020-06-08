# Adapted 'produceOtherDataStructure.py' to save the image paths to a file.
# The file is a set of subjects, ach subject is a list of:
# 1. subject name 'name' which is a string;
# 2. list subject slices 'slices', which is a list of:
#    i.   'img1'
#    ii.  'img2'
#    iii. 'structure'
#    iv.  'lesion' if it is a patient
#    v.   info if it is a patient 'is_patient' which is boolean
# 3. subject info if it is a patient 'is_patient' which is boolean

import os  # to read the folder structure
# import shutil  # to copy a file
# import warnings  # to throw a warning
import json  # working with files

data_name = 'data_final'
print('saving the data paths to data_mialab/'+data_name+'.txt')

# specify the path of your current data that you want to read
############################################ change here
readPath = 'data_mialab/testOnsite_public'

slices = []

# loop over each file in path recursively
for r, d, f in os.walk(readPath):  # r = root = current parent folder, d = current directorys (empty if there are no subdirectories in the current parent folder), f = current files (empty if there are no files in the current parent folder, e.g. if the parent folder is empty or if there are only subdirectories)

    structureArray = r.split(os.path.sep)

    # saving all image paths in their corresponding slices
    if 'img1.nii.gz' in f and 'img2.nii.gz' in f:
        sli = {'img1': os.path.join(r, "img1.nii.gz"),
               'img2': os.path.join(r, "img2.nii.gz")}

        # saving the subject name
        for infoPart in structureArray:
            if 'subject' in infoPart:
                sli['subject'] = infoPart
                break

        # saving the slice name since some subjects have many slices with same slice name
        sli['slice'] = r

        slices.append(sli)

    else:
        for file in f:
            if '.nii.gz' in file:
                print('Found nii-file outside of slice: ', file)

# saving the subject names
subjects = []
subject_names = set()
for sli in slices:
    subject_names.add(sli['subject'])

# saving the subject name in subjects
for s in subject_names:
    subjects.append({'name': s, 'slices': []})

# saving the slices in subjects
for sub in subjects:
    for sli in slices:
        if sli['subject'] == sub['name']:
            sub['slices'].append(sli)

# save the slice paths in a file
json.dump(slices, open('data_mialab/'+data_name+'.txt', 'w'))  # subject-wise

print('done!')
