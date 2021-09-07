import json
from os import path, listdir

with open("research_archive/datasets/YoutubeVOS/valid.json", "r") as f:
    data = json.load(f)
data = data["videos"]

vids = listdir("research_archive/datasets/valid_all_frames/JPEGImages")

for entry in data:
    name = entry["file_names"][0].split("/")[0]
    
    if entry in vids:
        print(name)