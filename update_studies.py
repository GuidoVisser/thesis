import json
import pprint as prettyprint

pp = prettyprint.PrettyPrinter(indent=2)
pprint = pp.pprint

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
    
        if not "times" in dicts[study]["configs"][setting].keys():
            dicts[study]["configs"][setting]["times"] = {video: t}
        else:
            dicts[study]["configs"][setting]["times"][video] = t
        
        if not "times" in dicts[study]["videos"][video].keys():
            dicts[study]["videos"][video]["times"] = {setting: t}
        else:
            dicts[study]["videos"][video]["times"][setting] = t

for study in studies:
    with open(f"job_scripts/experiments/{study}_updated.json", "w") as f:
        json.dump(dicts[study], f)