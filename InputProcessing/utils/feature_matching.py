import numpy as np
import matplotlib.cm as cm
import torch
from os import path
from typing import Union

from models.third_party.SuperGlue.models.matching import Matching
from models.third_party.SuperGlue.models.utils import make_matching_plot, AverageTimer

from utils.utils import create_dir

def get_model_config(superglue: str = "outdoor",
                     max_keypoints: int = 1024,
                     keypoint_threshold: float = 0.005,
                     nms_radius: int = 4,
                     sinkhorn_iterations: int = 20,
                     match_threshold: float = 0.2):

    assert superglue in ["outdoor", "indoor"], "Invalid name \'superglue\', choose from [\"outdoor\" \"indoor\"]"

    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }
    return config

@torch.no_grad()
def extract_and_match_features(frames: list,
                   device: str,
                   model_config: dict,
                   output_dir: Union[str, None] = None,
                   visualize: bool = False,
                   show_keypoints: bool = False,
                   interval: int = 1) -> dict:
    """
    """

    # Load the SuperPoint and SuperGlue models.
    matching = Matching(model_config).eval().to(device)
    # Create the output directories if they do not exist already.
    if output_dir is not None:
        create_dir(output_dir)

    timer = AverageTimer(newline=True)
    output = []
    for i in range(len(frames)-1):

        image0 = frames[i+1]
        image1 = frames[i]

        # Perform the matching.
        pred = matching({'image0': image0, 'image1': image1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1  = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        timer.update('feature matching')

        # Write the matches to disk.
        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                        'matches': matches, 'match_confidence': conf}
        output.append(out_matches)
        
        if output_dir is not None:
            # Get the save path
            matches_path = path.join(output_dir, f"{i*interval:05}_{(i+1)*interval:05}_matches.npz")
            
            # save points
            np.savez(str(matches_path), **out_matches)

        if visualize:
            # Get the save path
            viz_path = path.join(output_dir, f"{i*interval:05}_{(i+1)*interval:05}_matches.png")

            # Keep the matching keypoints.
            valid  = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf  = conf[valid]

            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                "SuperGlue",
                f"Keypoints: {len(kpts0)}:{len(kpts1)}"
                f"Matches: {len(mkpts0)}"
            ]

            # Display extra parameter info.
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                f"Keypoint Threshold: {k_thresh:.4f}",
                f"Match Threshold: {m_thresh:.2f}",
                f"Image Pair: {i*interval:05}:{(i+1)*interval:05}"
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, show_keypoints, False, 
                False, 'Matches', small_text)

            timer.update('visualize')

        # timer.print(f"Finished pair {i:5} of {len(frames)-2:5}") # -2:= -1 for zero index, -1 because we look at pairs i.s.o. frames

    return output

def get_matching_coordinates(keypoint_matches: list) -> list:
    """
    Given a set of keypoints in a pair of frames, return two sets of coordinates of the keypoints. 
    Keypoints that have been matched to each other are connected to each other by having the same index in the arrays
    
    Args:
        matches (list[dict]): list of dictionaries containing the pixel locations of keypoints in two disinct frames 
                                as well as the matchings between them

    Returns:
        coords (list[np.array]): list of pixel coordinates in the form
            [
                [
                    coords0 (np.array) : coordinates in the first frame of each pair
                    coords1 (np.array) : coordinates in the second frame of each pair
                ],
                [
                    ...
                ],
                ...
            ]
    """

    coords = []
    for pair in keypoint_matches:
        
        # get keypoints in both frames and matching indices
        keypoints0 = pair["keypoints0"]
        keypoints1 = pair["keypoints1"]
        matches    = pair["matches"]

        # get coordinates of matching pairs
        coords0 = []
        coords1 = []
        for i, point in enumerate(keypoints0):
            if matches[i] > -1:
                coords0.append(point)
                coords1.append(keypoints1[matches[i]])
        
        coords.append([
            np.array(coords0),
            np.array(coords1)
        ])

    return coords
