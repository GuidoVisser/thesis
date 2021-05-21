import torch
import argparse
from os import path
from datetime import datetime

from torchvision.utils import save_image

from models.TopkSTM.prop_net import PropagationNetwork
from models.TopkSTM.utils.utils import pad_divide_by, aggregate_wbg
from datasets import DAVISVideo
from utils.transforms import get_transforms
from utils.utils import create_dir
from utils.video_utils import create_masked_video

def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results_dir = path.join(args.log_dir, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    create_dir(results_dir)

    model = PropagationNetwork(top_k=args.top_k).to(device)
    model.load_state_dict(torch.load(args.model_path))

    dataset = DAVISVideo(args.data_dir, args.video, get_transforms())
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        pin_memory=True
    )

    mem_freq = args.mem_freq
    total_m = (len(dataset) - 1) // mem_freq + 2 # +1 for first frame, +1 to make sure indexing remains within bounds

    frame, mask = next(iter(dataloader))
    frame, mask = frame.to(device), mask.to(device)

    frame, _ = pad_divide_by(frame, 16)
    mask, _ = pad_divide_by(mask, 16)
    print(torch.mean(mask))
    print(torch.std(mask))


    key, value = model.memorize(frame, mask)

    K, CK, _, H, W = key.size()
    _, CV, _, _, _ = value.size()

    # Pre-allocate keys/values memory
    keys = torch.empty((K, CK, total_m, H, W), dtype=torch.float32, device=device)
    values = torch.empty((K, CV, total_m, H, W), dtype=torch.float32, device=device)

    m_front = 1
    keys[:, :, :m_front] = key
    values[:, :, :m_front] = value

    for i, (frame, _) in enumerate(dataloader):
    
        if i == 0:
            continue

        frame = frame.to(device)
        frame, _ = pad_divide_by(frame, 16)

        key = keys[:, :, :m_front]
        value = values[:, :, :m_front]
        query = model.get_query_values(frame)

        mask_pred = model.segment_with_query(key, value, *query)
        mask_pred = aggregate_wbg(mask_pred)

        save_image(mask_pred, path.join(results_dir, f"{i:05}.png"))

        new_k, new_v = model.memorize(frame, mask_pred)
        if (i+1) % mem_freq == 0:
            m_front += 1
        keys[:, :, m_front:m_front+1] = new_k
        values[:, :, m_front:m_front+1] = new_v

    create_masked_video(f"{args.data_dir}/480p/JPEGImages/{args.video}", results_dir, save_path=path.join(results_dir, "demo.mp4"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="datasets/DAVIS_minisample")
    parser.add_argument("--video", type=str, default="tennis")
    parser.add_argument("--log_dir", type=str, default="results/topkSTM")
    parser.add_argument("--model_path", type=str, default="models/weights/MiVOS/propagation_model.pth")

    parser.add_argument("--mem_freq", type=int, default=5, help="Frequency at which to expand the memory")
    parser.add_argument("--top_k", type=int, default=50, help="top k channels of attention are used to reduce noise in the output")

    args = parser.parse_args()

    main(args)