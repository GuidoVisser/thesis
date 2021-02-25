import json

with open("datasets/YoutubeVOS/train.json", "r") as f:
    data = json.load(f)

print(data.keys())

videos = data["videos"]
video = next((vid for vid in videos if vid["id"] == 23))

annotations = data["annotations"]

relevant_anns = [ann for ann in annotations if ann["video_id"] == 23]

data["videos"] = [video]
data["annotations"] = relevant_anns

with open("datasets/YoutubeVOS_sample/valid/valid.json", "w") as f:
    json.dump(data, f)

with open("datasets/YoutubeVOS_sample/annotations/instances_valid.json", "w") as f:
    json.dump(relevant_anns, f)