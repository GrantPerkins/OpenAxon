import json
import sys
from glob import glob
from os import path
import cv2
import pandas
from openimages.download import download_dataset

data_json = sys.argv[1]
assert path.isfile(data_json)
with open(data_json) as file:
    data = json.load(file)
labels = data["labels"]
assert type(labels) == list
limit = data["limit"]
assert type(limit) == int

print("Getting dataset, size: {}, contents: {}".format(limit, labels))

directory = "./data/train"
# download_dataset(dest_dir=directory, csv_dir=directory, class_labels=["Segway"], annotation_format="pascal", exclusions_path=None, limit=10)
images = glob("./data/train/*/images/*.jpg")
label_map_df = pandas.read_csv(path.join(directory, "class-descriptions-boxable.csv"))
label_map = {}

with open(path.join(directory, "class-descriptions-boxable.csv")) as file:
    for row in file.readlines():
        row = row.rstrip().split(',')
        label_map.update({row[0]: row[1]})


csv = {}
for image_path in images:
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    file_id = image_path.split("/")[-1].rstrip(".jpg")
    csv.update({file_id: {"height": height, "width": width, "labels": []}})

all_labels_df = pandas.read_csv(path.join(directory, "train-annotations-bbox.csv"))
all_labels_df.set_index("ImageID", inplace=True)
print(len(csv.keys()))
for key in csv.keys():
    row = all_labels_df.loc[key]
    print(type(row))
    for i in row:
        print(i)
    print()


# for index, row in all_labels_df.iterrows():
#     image_id = row["ImageID"]
#     if image_id in csv.keys():
#         label = label_map[row["LabelName"]]
#         csv[image_id]["labels"].append({"label": label, "xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0})
# print(csv)
