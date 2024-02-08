import os
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from sklearn.model_selection import KFold
import random
import numpy as np

random_seed_list = [42, 52, 62, 72, 82,]
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

# Assuming you have 'Eye ID' and 'Final Label' columns in the DataFrame
image_path_and_label_dataframe = image_path_and_label_dataframe[[
    'Eye ID', 'Final Label']]
class_name = ["NRG", "RG"]
class_to_numeric = {class_label: idx for idx,
                    class_label in enumerate(class_name)}
# numeric_labels = [class_to_numeric[label]
#                   for label in labels]
# Split DataFrame into two based on the 'Column_Name' values
nrg_index = image_path_and_label_dataframe.index[image_path_and_label_dataframe['Final Label'] == 'NRG'].tolist(
)

rg_index = image_path_and_label_dataframe.index[image_path_and_label_dataframe['Final Label'] == 'RG'].tolist(
)

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
                   shuffle=True, random_state=random_seed)
        random.seed(random_seed)  # Set the random seed
        input_paths = image_path_and_label_dataframe["Eye ID"]
        labels = image_path_and_label_dataframe["Final Label"]
        all_splits = [k for k in kf.split(
            input_paths, labels)]

        train_indexes, val_indexes = all_splits[k]

        geo_images = [f for f in os.listdir(
            geo_aug_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        color_images = [f for f in os.listdir(
            color_aug_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        train_dataframe = image_path_and_label_dataframe.iloc[train_indexes, :]
        val_dataframe = image_path_and_label_dataframe.iloc[train_indexes, :]
        train_nrg_index = train_dataframe.index[
            train_dataframe['Final Label'] == 'NRG']
        train_rg_index = train_dataframe.index[
            train_dataframe['Final Label'] == 'RG']
        val_nrg_index = val_dataframe.index[
            val_dataframe['Final Label'] == 'NRG']
        val_rg_index = val_dataframe.index[val_dataframe['Final Label'] == 'RG']

        train_rg_count = len(train_rg_index.tolist()) + \
            len(geo_images) + len(color_images)
        val_rg_count = len(val_rg_index.tolist())
        train_nrg_count = len(train_nrg_index.tolist())
        val_nrg_count = len(val_nrg_index.tolist())
        print("train_augmented/class_ones_count: ", train_rg_count)
        print("train_augmented/class_zeros_count: ", train_nrg_count)
        print("val_augmented/class_ones_count: ", val_rg_count)
        print("val_augmented/class_zeros_count: ", val_nrg_count)
        # Use random.sample to get train_rg_count random indices from ngr_indices
        train_nrg_selected_indices = random.sample(
            train_nrg_index.tolist(), train_rg_count)
        val_nrg_selected_indices = random.sample(
            val_nrg_index.tolist(), val_rg_count)
        # Get the corresponding input and label data using the selected indices
        train_selected_nrg_data = image_path_and_label_dataframe.iloc[
            train_nrg_selected_indices, :]
        val_selected_nrg_data = image_path_and_label_dataframe.iloc[val_nrg_selected_indices, :]
        train_rg_input_data = image_path_and_label_dataframe.iloc[train_rg_index, :]
        val_rg_input_data = image_path_and_label_dataframe.iloc[val_rg_index, :]
        train_combined_nrg_input_data = pd.concat(
            [train_selected_nrg_data, train_rg_input_data], ignore_index=True)
        val_combined_nrg_input_data = pd.concat(
            [val_selected_nrg_data, val_rg_input_data], ignore_index=True)
# rg_df = [class_to_numeric[rg]
#          for rg in nrg_df]
# images = []
# for path in input_paths:
#     image = Image.open(fp=path, mode='r')
#     # Convert RGBA to RGB
#     image = image.convert('RGB')
#     images.append(image)
