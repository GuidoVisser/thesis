import torchvision
import torch

import transforms as T
import utils
from engine import train_one_epoch, evaluate

from datasets import TestVideoDataset

from time import time
import os


def get_transforms(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset_test = TestVideoDataset("test_dataset", get_transforms(train=False))
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device)

    model.eval()
    imgs, targets = next(iter(data_loader_test))
    imgs = [img.to(device) for img in imgs]
    
    for i, (imgs, targets) in enumerate(data_loader_test):
        imgs = [img.to(device) for img in imgs]
        print(imgs[0].shape)


        # targets = [target.to(device) for target in targets]

        t = time()
        with torch.no_grad():
            pred = model(imgs)
            masks = pred[0]["masks"]
            
            print("masks generated: ", time() - t)

            for j, mask in enumerate(masks):
                if not os.path.exists(f"test_output/{j:02d}"):
                    os.mkdir(f"test_output/{j:02d}")

                mask = torch.where(mask > 0.1, 1., 0.)

                torchvision.utils.save_image(mask, f"test_output/{j:02d}/{i:02d}.png")

            print("saved masks: ", time() - t)
        
    print("party")

if __name__ == "__main__":
    main()