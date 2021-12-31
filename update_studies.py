import json
from math import ceil

dicts = {}

studies = [study + "_study" for study in ["3d_conv", "noise", "attention", "alpha", "depth", "optical_flow"]]

for study in studies:
    with open(f"job_scripts/experiments/{study}.json", "r") as f:
        dicts[study] = json.load(f)

with open("times.txt") as f:
    data = f.readlines()

for line in data:
    line = line.strip().split(" ")
    if len(line) == 4:
        _, video, identifier, t = line

        study, setting = identifier.split("__")
    
        if not "sec_per_epoch" in dicts[study]["configs"][setting].keys():
            dicts[study]["configs"][setting]["sec_per_epoch"] = {video: ceil(float(t))}
        else:
            dicts[study]["configs"][setting]["sec_per_epoch"][video] = ceil(float(t))
    else:
        print(line)

for study in studies:
    with open(f"job_scripts/experiments/{study}.json", "w") as f:
        json.dump(dicts[study], f, indent=4) 