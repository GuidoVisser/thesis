from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from os import path
import json
import numpy as np
import matplotlib.pyplot as plt

data_dir = "datasets/YoutubeVOS_sample"

ann_file = path.join(data_dir, "valid/result.json")

with open(ann_file, "r") as af:
    data = json.load(af)

annotations = data["annotations"]

coco = COCO()

for ann in annotations:
    print(ann)

    counts = ann["segmentations"][0]["counts"]
    height = ann["segmentations"][0]["size"][0]
    width = ann["segmentations"][0]["size"][1]
    print(width)
    rle = maskUtils.frPyObjects(ann["segmentations"], width, height)
    mask = maskUtils.decode(rle)
    plt.imshow(mask[:,:,1])
    plt.show()