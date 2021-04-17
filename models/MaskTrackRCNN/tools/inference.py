import torch
import mmcv
import json
import numpy as np
from os import path, listdir
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from pycocotools.coco import COCO

from mmdet import datasets

from mmdet.apis import inference_detector, show_result
from mmdet.models import build_detector, detectors
from mmdet.core import results2json_videoseg

with open(path.join(root_path, "data/demo/drive-download-20210222T152934Z-001/valid.json"), "r") as f:
    dataset = json.load(f)

root_path = path.dirname(__file__) + "/../"
cfg = mmcv.Config.fromfile(path.join(root_path, "configs/masktrack_rcnn_r50_fpn_1x_youtubevos.py"))
checkpoint_path = path.join(root_path, "epoch_12.pth")
image_folder = path.join(root_path, "data/valid/JPEGImages/0e4068b53f")
result_path = path.join(root_path, "results.json")
imgs = [path.join(image_folder, f) for f in listdir(image_folder) if path.isfile(path.join(image_folder, f))]

model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
load_checkpoint(model, checkpoint_path)


results = inference_detector(model, imgs, cfg)
collected_results = []
for r in results:
    collected_results.append(r[0])

for vid in dataset["videos"]:
    if vid["id"] == 23:
        break
print(vid)
