import json
from math import ceil

n_epochs = 500
dicts = {}

studies = [study + "_study" for study in ["3d_conv", "noise", "attention", "alpha", "depth", "optical_flow"]]

for study in studies:
    with open(f"job_scripts/experiments/{study}.json", "r") as f:
        dicts[study] = json.load(f)

for study in dicts.keys():
    for experiment in dicts[study]["configs"].keys():
        dicts[study]["configs"][experiment]["n_epochs"] = n_epochs
    #     del dicts[study]["configs"][experiment]["time_estimate"]
        
    # for video in dicts[study]["videos"].keys():
    #     del dicts[study]["videos"][video]["length_modifier"]

for study in studies:
    with open(f"job_scripts/experiments/{study}.json", "w") as f:
        json.dump(dicts[study], f, indent=4) 