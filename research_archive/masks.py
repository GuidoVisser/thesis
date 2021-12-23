from torchvision.models.detection import maskrcnn_resnet50_fpn
from InputProcessing.frameIterator import FrameIterator

import cv2

if __name__ == "__main__":

    fi = FrameIterator("datasets/Jaap_Jelle/JPEGImages/480p/nescio_1", frame_size=(448, 256))

    input = [frame for frame in fi]

    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    predictions = model(input[0:4])

    for i, frame in enumerate(predictions):
        for j in range(frame["masks"].shape[0]):
            mask = frame["masks"][j].permute(1, 2, 0).detach().numpy()
            img  = input[i].permute(1, 2, 0).numpy() 

            demo_img = (0.5 + 2* mask) * img

            cv2.imshow(f"{i}_{j}_{frame['labels'][j]}_test", demo_img)
            cv2.imshow(f"{i}_{j}_{frame['labels'][j]}_mask", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

