from InputProcessing.inputProcessor import InputProcessor

if __name__ == "__main__":
    video = "datasets/DAVIS/JPEGImages/480p/tennis"
    mask_dir = "datasets/DAVIS/Annotations/480p/tennis"
    flow_dir = "results/flow_testing"

    ip = InputProcessor(video, mask_dir, flow_dir)
    ip.get_frame_input(40)