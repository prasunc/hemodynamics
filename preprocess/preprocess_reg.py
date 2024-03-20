# -*- coding: utf-8 -*-




import pandas as pd
import numpy as np

import torchvision.transforms as transforms
from skimage.transform import estimate_transform, rescale, warp
import pydicom

import matplotlib.pyplot as plt
from kale.interpret import visualize
#from geodesic import geodesic_smooth_stack
import cv2



def read_dicom_phases(dicom_path, sort_instance=True):
    """Read dicom images of multiple instances/phases for one patient.
    Args:
        dicom_path (str): Path to DICOM images.
        sort_instance (bool, optional): Whether sort images by InstanceNumber (i.e. phase number). Defaults to True.
    Returns:
        [list]: List of dicom dataset objects
    """
    dcm_phases = []  # list of dicom dataset objects (phases)
    # get all dicom files under the directory
    phase_files = glob.glob(dicom_path + "/**/*.dcm", recursive=True)
    for phase_file in phase_files:
        dataset = pydicom.dcmread(phase_file)
        setattr(dataset, "FilePath", phase_file)
        dcm_phases.append(dataset)
    if sort_instance:
        dcm_phases.sort(key=lambda x: x.InstanceNumber, reverse=False)

    return dcm_phases


def dicom2arraylist(dicom_patient_list, return_patient_id=False):
    """Convert dicom datasets to arrays
    Args:
        dicom_patient_list (list): List of dicom patient lists.
        return_patient_id (bool, optional): Whether return PatientID. Defaults to False.
    Returns:
        list: list of array-like tensors.
        list (optional): list of PatientIDs.
    """
    n_samples = len(dicom_patient_list)
    patient_ids=dicom_patient_list[0].PatientID
    image_list = []  # number of phases can be different across patients, using list to avoid the phase dimension issue
    for i in range(n_samples):
        img = dicom_patient_list[i].pixel_array
        image_list.append(img)
    if return_patient_id:
        return image_list, patient_ids
    else:
        return image_list
    
def reg_img_stack(images, coords, target_coords):
    """Registration for stacked images

    Args:
        images (list): Input data, where each sample in shape (n_phases, dim1, dim2).
        coords (array-like): Coordinates for registration, shape (n_samples, n_landmarks * 2).
        target_coords (array-like): Target coordinates for registration.

    Returns:
        list: Registered images, each sample in the list in shape (n_phases, dim1, dim2).
        array-like: Maximum distance of transformed source coordinates to destination coordinates, shape (n_samples,)
    """
    n_samples = len(images)
    if n_samples != coords.shape[0]:
        error_msg = "The sample size of images and coordinates does not match."
        logging.error(error_msg)
        raise ValueError(error_msg)
    n_landmarks = int(coords.shape[1] / 2)

    target_coords = target_coords.reshape((n_landmarks, 2))
    max_dist = np.zeros(n_samples)
    for i in range(n_samples):
        src_coord = coords[i, :]
        src_coord = src_coord.reshape((n_landmarks, 2))
        idx_valid = np.isnan(src_coord[:, 0])
        tform = estimate_transform(ttype="similarity", src=src_coord[~idx_valid, :], dst=target_coords[~idx_valid, :])
        # forward transform used here, inverse transform used for warp
        src_tform = tform(src_coord[~idx_valid, :])
        dists = np.linalg.norm(src_tform - target_coords[~idx_valid, :], axis=1)
        max_dist[i] = np.max(dists)
        n_phases = len(images[i])
        for j in range(n_phases):
            src_img = images[i][j].copy()
            warped = warp(src_img, inverse_map=tform.inverse, preserve_range=True)
            images[i][j] = warped

    return images, max_dist
    

def rescale_img_stack(images, y, scale=0.5):
    """Rescale stacked images by a given factor
    Args:
        images (list): Input data list, where each sample in shape (n_phases, dim1, dim2).
        scale (float, optional): Scale factor. Defaults to 0.5.
    Returns:
        list: Rescaled images, each sample in the list in shape (n_phases, dim1 * scale, dim2 * scale).
    """
    n_samples = len(images)
    # n_phases = images.shape[:2]
    images_rescale = []
    for i in range(n_samples):
        stack_i = []
        n_phases = len(images[i])
        for j in range(n_phases):
            img = images[i][j]
            img_rescale = rescale(img, scale, preserve_range=True)
            # preserve_range should be true otherwise the output will be normalised values
            stack_i.append(img_rescale.reshape((1,) + img_rescale.shape))
        stack_i = np.concatenate(stack_i, axis=0)
        images_rescale.append(stack_i)

    return images_rescale   


def mask_img_stack(images, mask):
    """Masking stacked images by a given mask

    Args:
        images (list): Input image data, where each sample in shape (n_phases, dim1, dim2).
        mask (array-like): mask, shape (dim1, dim2).
    Returns:
        list: masked images, each sample in the list in shape (n_phases, dim1, dim2).
    """
    n_samples = len(images)
    for i in range(n_samples):
        n_phases = len(images[i])
        for j in range(n_phases):
            images[i][j] = np.multiply(images[i][j], mask)

    return images


def normalize_img_stack(images):
    """Normalize pixel values to (0, 1) for stacked images.

    Args:
        images (list): Input data, where each sample in shape (n_phases, dim1, dim2).

    Returns:
        list: Normalized images, each sample in the list in shape (n_phases, dim1, dim2).
    """
    n_samples = len(images)
    for i in range(n_samples):
        n_phases = len(images[i])
        for j in range(n_phases):
            img = images[i][j]
            images[i][j] = (img - np.min(img)) / (np.max(img) - np.min(img))

    return images


#read template mask
mask=cv2.imread('mask1.png',0)

landmark_train=pd.read_csv('training_shortaxis.csv',index_col='patient_id')
list_of_path_train=landmark_train["image_path"].values
landmark_train=landmark_train.iloc[:,:-1].values

#select a target scan for registration

target=landmark_train[20]

landmark_test=pd.read_csv('training_shortaxis.csv',index_col='patient_id')
list_of_path_test=landmark_test["image_path"].values
landmark_test=landmark_test.iloc[:,:-1].values

# load training scans
train_images=[]
train_pids=[]
for i in range(len(list_of_path_train)):
    disc=read_dicom_phases(list_of_path_train[i],sort_instance=True)
    img,ptd=dicom2arraylist(disc.copy(),return_patient_id=True)
    train_images.append(img)
    train_pids.append(ptd)


#load testing scans

test_images=[]
test_pids=[]
for i in range(len(list_of_path_test)):
    disc=read_dicom_phases(list_of_path_test[i],sort_instance=True)
    img,ptd=dicom2arraylist(disc.copy(),return_patient_id=True)
    test_images.append(img)
    test_pids.append(ptd)



'''
preprocessing training samples
'''

#perform registration
training_img_reg, max_dist = reg_img_stack(train_images.copy(), landmark_train, target)

# Mask the scans
training_img_masked = mask_img_stack(training_img_reg.copy(), mask)

#Normalize scans

training_img_norm = normalize_img_stack(training_img_masked.copy())

# Rescale scans to create different resolutions such as 128X28, 64X64, etc.

training_img_rescaled = rescale_img_stack(training_img_norm.copy(), train_labels, scale=0.125)




'''
preprocessing testing samples
'''

#perform registration on testing scans

testing_img_reg, max_dist = reg_img_stack(test_images.copy(), landmark_test, target)


#perform masking on test set

testing_img_masked = mask_img_stack(testing_img_reg.copy(), mask)

#perform normalization on test set

testing_img_norm = normalize_img_stack(testing_img_masked.copy())

#rescale scans

testing_img_rescaled = rescale_img_stack(testing_img_norm.copy(), test_labels, scale=0.125)




#x_train = np.concatenate([training_img_rescaled[i].reshape((1,) + training_img_rescaled[i].shape) for i in range(len(training_img_rescaled))], axis=0)
#x_test = np.concatenate([testing_img_rescaled[i].reshape((1,) + testing_img_rescaled[i].shape) for i in range(len(testing_img_rescaled))], axis=0)


#Save Preprocessed data
np.save('sxtrain.npy',x_train, allow_pickle=True)
np.save('sxtest.npy',x_test, allow_pickle=True)


