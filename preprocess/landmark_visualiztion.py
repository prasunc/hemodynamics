# -*- coding: utf-8 -*-


import os
import matplotlib.pyplot as plt
import pandas as pd
from config import get_cfg_defaults
from kale.loaddata.image_access import read_dicom_dir
from visualize import plot_multi_images
import pydicom


cfg_path = "tutorial_svc.yaml" # Path to `.yaml` config file

cfg = get_cfg_defaults()
cfg.merge_from_file(cfg_path)
cfg.freeze()
print(cfg)

base_dir = cfg.DATASET.BASE_DIR
file_format = cfg.DATASET.FILE_FORAMT

landmark='fc_landmarks.csv'
landmark_file=pd.read_csv(landmark, index_col="subject")
landmark1=landmark_file.iloc[:, :-1].values
list_of_paths=landmark_file["image_path"].values

image_list=[]


for i in range(len(list_of_paths)):
    dicom=pydicom.dcmread(path[i])
    array=dicom.pixel_array
    image_list.append(array)

#landmark_file1=landmark_file["Inf insertion x", "Inf insertion y", "Mid LV lateral wall x", "Mid LV lateral wall y", "RV insertion x", "RV insertion y","Up insertion x","Up insertion y"].values
#x=landmark_file.columns[5:12:2].values
marker_names = list(landmark_file.columns[1:7:2])
markers = []
for marker in marker_names:
    marker_name = marker.split(" ")
    marker_name.pop(-1)
    marker_name = " ".join(marker_name)
    markers.append(marker_name)
    
    
plot_multi_images(
    [image_list[i] for i in range(len(image_list))],
    marker_locs=landmarks1,
    im_kwargs=dict(cfg.PLT_KWS.IM),
    marker_titles=markers,
    marker_cmap="Set1",
    marker_kwargs=dict(cfg.PLT_KWS.MARKER),
    n_cols=5
).show()
    

    
