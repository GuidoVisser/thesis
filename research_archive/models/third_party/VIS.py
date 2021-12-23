import torch
import mmcv
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image
from os import path, listdir, mkdir
from mmcv.runner import load_checkpoint
from pycocotools import mask as maskUtils

from models.third_party.MaskTrackRCNN.mmdet.apis import inference_detector, show_result
from models.third_party.MaskTrackRCNN.mmdet.models import build_detector, detectors

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class MaskGenerator(object):

    def __init__(self, data_dir, results_dir, model_config, model_checkpoint="MaskTrackRCNN/epoch_12.pth"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.cfg = model_config
        self.model = build_detector(self.cfg.model, train_cfg=None, test_cfg=self.cfg.test_cfg)
        load_checkpoint(self.model, model_checkpoint)

    def generate_annotations_for_video(self, video, start_id=0):
        """
        Generate annotations for each object in a given video

        Args:
            video (dict): coco style video dict with information on the video
            start_id (int): the id at which to start counting
        
        Returns:
            annotations (list): A coco style annotation dict for the given video
        """

        frames = video["file_names"]
        frame_paths = [path.join(self.data_dir, "JPEGImages", frame) for frame in frames] 
    
        results = inference_detector(self.model, frame_paths, self.cfg)

        annotations = {}
        all_ids = set([])
        for i, (bboxes, segmentations) in enumerate(results):
            current_ids = set([])
            for j in bboxes.keys():
                j = int(j)

                all_ids.add(j)
                current_ids.add(j)

                if not j in annotations.keys():
                    annotations[j] = {
                        "id": start_id + j,
                        "video_id": video["id"],
                        "category_id": bboxes[j]["label"],
                        "iscrowd": 0,
                        "segmentations": [None]*i + [{"counts":list(segmentations[j]["counts"]), "size":segmentations[j]["size"]}],
                        "bboxes": [None]*i + [list(bboxes[j]["bbox"])],
                        "areas": [None]*i + [0], #TODO calculate area
                        "width": segmentations[j]["size"][1],
                        "height": segmentations[j]["size"][0],
                        "length": 1
                    }
                else:
                    annotations[j]["segmentations"].append({"counts":list(segmentations[j]["counts"]), "size":segmentations[j]["size"]})
                    annotations[j]["bboxes"].append(list(bboxes[j]["bbox"]))
                    annotations[j]["areas"].append(0) #TODO calculate area

            for idx in all_ids:
                if not idx in current_ids:
                    annotations[idx]["segmentations"].append(None)
                    annotations[idx]["bboxes"].append(None)
                    annotations[idx]["areas"].append(None) #TODO calculate area

        annotations = [annotations[ann] for ann in sorted(annotations.keys())]
        return annotations

    # def generate_annotations_for_single_video(self, video_id):
    #     """
    #     Generate a mask for a specific video

    #     Args:
    #         video_id (int): coco style id for video

    #     returns:
    #         annotations (list): coco style annotations for the video
    #     """
    #     with open(path.join(self.data_dir, "train.json"), "r") as f:
    #         data = json.load(f)
    #     videos = data["videos"]
    #     video = next((vid for vid in videos if vid["id"] == video_id))
    #     annotations = self.generate_annotations_for_video(video)

    #     return video, annotations

    def generate_annotations(self, data_path, result_path):
        """
        Generate the annotations for the given data set
        """
        with open(path.join(self.data_dir, data_path), "r") as f:
            data = json.load(f)

        data["annotations"] = []

        videos = data["videos"]
        for video in videos:
            start_id = len(data["annotations"])
            data["annotations"].extend(self.generate_annotations_for_video(video, start_id))

        with open(path.join(self.data_dir, result_path), "w") as f:
            json.dump(data, f, cls=NpEncoder)    

    def generate_masks_from_video_dir(self, data_path):
        """
        Generate a set of masks for the given video
        """
        print(data_path)
        print(self.results_dir)
        if not path.exists(path.join(self.results_dir, data_path)):
            mkdir(path.join(self.results_dir, data_path))

        frame_paths = sorted([path.join(self.data_dir, data_path, frame) for frame in listdir(path.join(self.data_dir, data_path))])
    
        results_generator = inference_detector(self.model, frame_paths, self.cfg)
        
        max_k = 0
        for i, (bboxes, segmentations) in enumerate(results_generator):
            for k in segmentations.keys():

                if k > max_k:
                    max_k = k

                current_path = path.join(self.results_dir, data_path, f"id_{k}")
                if not path.exists(current_path):
                    mkdir(current_path)

                mask = maskUtils.decode(segmentations[k])
                mask = self.dilate_mask(mask)
                
                save_path = path.join(current_path, f"{i:05}.png")
                plt.imsave(save_path, mask, cmap=cm.gray)

        meta_data = {
            "frame_count": len(frame_paths), 
            "n_instances": max_k,
            "shape": mask.shape 
            }

        with open(path.join(self.results_dir, data_path, "info.json"), "w") as f:
            json.dump(meta_data, f, cls=NpEncoder)


    def dilate_mask(self, mask, dilation=15):
        """
        Dilate the binary mask by a given amount of pixels

        Args:
            mask (torch.Tensor): binary mask
            dilatation (int): amount of pixels that the mask needs to be dilated (defaults to 15 per Gao et al.)
        
        Returns:
            dilated_mask (torch.Tensor): dilated binary mask
        """
        kernel = np.ones((dilation, dilation))
        dilated_mask = cv2.dilate(mask, kernel)
        
        return dilated_mask

    def combine_all_masks(self, mask_dir):
        """
        Combine all instance masks in a given directory into one

        Args:
            mask_dir (str): directory where all masks are stored
        """
        if not path.exists(path.join(self.results_dir, mask_dir, "combined")):
            mkdir(path.join(self.results_dir, mask_dir, "combined"))

        with open(path.join(self.results_dir, mask_dir, "info.json"), "r") as f:
            meta_data = json.load(f)
        
        frame_count = meta_data["frame_count"]
        n_instances = meta_data["n_instances"]
        mask_shape = meta_data["shape"]

        for frame in range(frame_count):
            combined_mask = np.zeros(mask_shape)

            for instance in range(n_instances):
                if path.exists(path.join(self.results_dir, mask_dir, f"id_{instance}/{frame:05}.png")):
                    instance_mask = np.array(Image.open(path.join(self.results_dir, mask_dir, f"id_{instance}/{frame:05}.png")).convert("L"))
                    combined_mask = np.maximum(combined_mask, instance_mask)

            plt.imsave(path.join(self.results_dir, mask_dir, f"combined/{frame:05}.png"), combined_mask, cmap=cm.gray)       
            

if __name__ == "__main__":
    
    cfg = mmcv.Config.fromfile(path.join(path.dirname(__file__), "MaskTrackRCNN/configs/masktrack_rcnn_r50_fpn_1x_youtubevos.py"))
    data_dir = "datasets/DAVIS/JPEGImages/1080p/"
    results_dir = "results/MaskTrackRCNN_masks/DAVIS/1080p/"

    MG = MaskGenerator(data_dir, results_dir, cfg)

    MG.generate_masks_from_video_dir("tennis")
    

    
