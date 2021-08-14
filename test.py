from os import rename, path, listdir, remove

root = "results/layer_decomposition/test_new"
# dirs = ["masks/00", "images", "flow/forward/flow", "flow/confidence"]
# dirs = [path.join(root, dir) for dir in dirs]

# for dir in dirs:
#     for i, fp in enumerate(sorted(listdir(dir))):
#         ext = fp.split(".")[1]
#         old_name = path.join(dir, fp)
#         new_name = path.join(dir, f"{i:05}.{ext}")

#         print(old_name)
#         print(new_name)

for fp in sorted(listdir(path.join(root, "images"))):
    if len(fp.split(".")[0]) == 4:
        remove(path.join(root, "images", fp))