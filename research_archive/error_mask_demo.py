from os import listdir, path

from utils.video_utils import create_masked_video, load_frame, save_frame
from utils.mask_utils import generate_error_mask
from utils.utils import create_dir

if __name__ == "__main__":

    mask_dir = "datasets/DAVIS_sample_tennis/Annotations/480p/tennis"
    vid_dir = "datasets/DAVIS_sample_tennis/JPEGImages/480p/tennis"
    pred_dir = "results/SeqMaskPropVAE/2021_04_30_10_41_50/demo/epoch_10"
    save_dir = "results/SeqMaskPropVAE/2021_04_30_10_41_50/demo/errors"

    create_dir(save_dir)

    for mask in sorted(listdir(mask_dir)):
        gt = load_frame(path.join(mask_dir, mask))
        pred = load_frame(path.join(pred_dir, mask))

        error_mask = generate_error_mask(pred, gt)

        save_frame(error_mask, path.join(save_dir, mask))