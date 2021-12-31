import json

dicts = {}

studies = [study + "_study" for study in ["3d_conv", "noise", "attention", "alpha", "depth", "optical_flow"]]

for study in studies:
    with open(f"job_scripts/experiments/{study}.json", "r") as f:
        dicts[study] = json.load(f)

with open("times.txt") as f:
    data = f.readlines()

for line in data:
    if len(line == 4):
        print(line)
    else:
        print(line, "XXXX")