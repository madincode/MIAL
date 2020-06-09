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
import shutil  # to copy a file
import warnings  # to throw a warning
import json  # working with files

data_name = 'data_subjects'
print('saving the data paths to data_mialab/'+data_name+'.txt')

# specify the path of your current data that you want to read
readPath = 'data_mialab/train_public'

slices = []

# loop over each file in path recursively
for r, d, f in os.walk(readPath):  # r = root = current parent folder, d = current directorys (empty if there are no subdirectories in the current parent folder), f = current files (empty if there are no files in the current parent folder, e.g. if the parent folder is empty or if there are only subdirectories)

    # if you are not sure, what r, d, and f are, you can print them out by uncomenting the following 4 lines:
    # print(r)
    # print(d) # slices
    # print(f) # files
    # print(' ')

    structureArray = r.split(os.path.sep)

    # saving all image paths in their corresponding slices
    if 'img1.nii.gz' in f and 'img2.nii.gz' in f and 'structure.nii.gz' in f:
        sli = {'img1': os.path.join(r, "img1.nii.gz"),
               'img2': os.path.join(r, "img2.nii.gz"),
               'structure': os.path.join(r, "structure.nii.gz")}
        if "lesion.nii.gz" in f:
            sli['lesion'] = os.path.join(r, "lesion.nii.gz")
        sli['is_patient'] = "/patients/" in r

        # saving the subject name
        for infoPart in structureArray:
            if 'subject' in infoPart:
                # subject = infoPart
                # print(subject)
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
    # if sli['subject'] not in subject_names:
    subject_names.add(sli['subject'])
# print(subject_names)
# print(len(subject_names))

# saving the subject name in subjects
for s in subject_names:
    subjects.append({'name': s, 'slices': []})

# saving the slices in subjects
for sub in subjects:
    for sli in slices:
        if sli['subject'] == sub['name']:
            sub['slices'].append(sli)
            sub['is_patient'] = sli['is_patient']

# the data has subjects ['name'], their slices ['slices'], patient or not ['is_patient'] 
# the slices have ['img1', 'img2', 'structure', 'lesion' if exists, 'is_patient'
# print(subjects[0]['is_patient']) 

# save the slice paths in a file
json.dump(subjects, open('data_mialab/'+data_name+'.txt', 'w'))  # slice-wise
json.dump(slices, open('data_mialab/data_slices.txt', 'w'))  # subject-wise

print('done!')

# can open the file like this
# data = json.load(open("data_mialab/data_names.txt"))
# len(data) # shows how many slices in the data
# data[400]['img1'] # can open specific file in the slice
# ps = [sli for sli in data if sli['is_patient']] # can save patients in separate list
# all(['lesion' in p for p in ps]) # check if all patients have lesions
