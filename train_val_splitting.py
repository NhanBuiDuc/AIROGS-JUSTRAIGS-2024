import os
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from sklearn.model_selection import KFold
import random
import numpy as np

random_seed_list = ["42", "52", "62", "72", "82",]
data_dir = "./data/"
train_image_path = os.path.join(
    data_dir, "AIROGS_2024", "preprocessed_images")
train_gt_path = os.path.join(
    data_dir, "AIROGS_2024", "JustRAIGS_Train_labels.csv")
geo_aug_images = os.path.join(
    data_dir, "AIROGS_2024", "geo_aug_images")
color_aug_images = os.path.join(
    data_dir, "AIROGS_2024", "color_aug_images")
output_dir = os.path.join(
    data_dir, "AIROGS_2024", "5kflod_split_images")
image_path_and_label_dataframe = pd.read_csv(train_gt_path, delimiter=';')

input_paths = image_path_and_label_dataframe['Eye ID']
labels = image_path_and_label_dataframe['Final Label']
class_name = ["NRG", "RG"]
class_to_numeric = {class_label: idx for idx,
                    class_label in enumerate(class_name)}
# numeric_labels = [class_to_numeric[label]
#                   for label in labels]
# Split DataFrame into two based on the 'Column_Name' values
nrg_index = labels.index[labels == 'NRG'].tolist()
rg_index = labels.index[labels == 'RG'].tolist()

nrg_count = len(nrg_index)
rg_count = len(rg_index)

print("All NRG samples: ", nrg_count)
print("All RG samples: ", rg_count)

# nrg_df = image_path_and_label_dataframe.loc[nrg_index]
# reduced_nrg_df = nrg_df[:rg_count*10]
# rg_count = len(rg_index)
for random_seed in random_seed_list:
    for k in range(0, 5):

        kf = KFold(n_splits=5,
                   shuffle=True, random_state=random_seed+1)
        random.seed(random_seed)  # Set the random seed
        all_splits = [k for k in kf.split(
            input_paths, labels)]

        train_indexes, val_indexes = all_splits[k]
        train_input_data = input_paths[train_indexes]
        train_label_data = labels[train_indexes]

        val_input_data = input_paths[val_indexes]
        val_label_data = labels[val_indexes]

        geo_images = [f for f in os.listdir(
            geo_aug_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        color_images = [f for f in os.listdir(
            color_aug_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        train_rg_index = train_input_data.index[train_indexes == 'RG']
        train_nrg_index = train_input_data.index[train_indexes == 'NRG']
        train_rg_count = len(train_rg_index.tolist()) + \
            len(geo_images) + len(color_images)
        print("augmented_train/class_ones_count: ", train_rg_count)
        # Use random.sample to get train_rg_count random indices from ngr_indices
        train_nrg_selected_indices = random.sample(
            train_nrg_index.tolist(), train_rg_count)

        # Get the corresponding input and label data using the selected indices
        train_selected_nrg_input_data = train_input_data[train_nrg_selected_indices]
        train_selected_nrg_label_data = train_label_data[train_nrg_selected_indices]
        train_rg_input_data = train_input_data[train_nrg_index]
        train_rg_label_data = train_label_data[train_nrg_index]
        trained_combined_nrg_input_data = train_selected_nrg_input_data.tolist() + \
            train_rg_input_data
        trained_combined_nrg_label_data = train_selected_nrg_label_data.tolist() + \
            train_rg_label_data

# rg_df = [class_to_numeric[rg]
#          for rg in nrg_df]
# images = []
# for path in input_paths:
#     image = Image.open(fp=path, mode='r')
#     # Convert RGBA to RGB
#     image = image.convert('RGB')
#     images.append(image)
