import os
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from sklearn.model_selection import KFold

random_seed_list = ["111", "222", "333", "444", "555",]
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

nrg_df = image_path_and_label_dataframe.loc[nrg_index]
reduced_nrg_df = nrg_df[:rg_count*10]
rg_count = len(rg_index)
for random_seed in random_seed:
    kf = KFold(n_splits=5,
               shuffle=True, random_state=self.kfold_seed)

    all_splits = [k for k in kf.split(
        input_data, labels_numeric)]

    train_indexes, val_indexes = all_splits[self.kfold_index]
# rg_df = [class_to_numeric[rg]
#          for rg in nrg_df]
# images = []
# for path in input_paths:
#     image = Image.open(fp=path, mode='r')
#     # Convert RGBA to RGB
#     image = image.convert('RGB')
#     images.append(image)
