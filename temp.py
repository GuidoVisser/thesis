from os import path, listdir, rename, remove

def reduce_fr(dirname):

    for i, fp in enumerate(sorted(listdir(dirname))):
        if i % 2 == 1:
            remove(path.join(dirname, fp))
            print(path.join(dirname, fp))
        else:
            idx = i // 2
            ext = fp[-4:]
            old = path.join(dirname, fp)
            new = path.join(dirname, f"{idx:05}" + ext)

            rename(old, new)
            print(old, new)

def remove_first_n_frames(dirname, n):
    for i, fp in enumerate(sorted(listdir(dirname))):
        if i < n:
            continue

        ext = fp[-4:]
        old = path.join(dirname, fp)
        new = path.join(dirname, f"{i - n:05}{ext}")

        # rename(old, new)
        print(old, new)


if __name__ == "__main__":
    root = "datasets/Videos"

    images = path.join(root, "Images")
    annotations = path.join(root, "Annotations")

    videos = listdir(images)

    for video in videos:
        reduce_fr(path.join(images, video))
        for layer in listdir(path.join(annotations, video)):
            reduce_fr(path.join(annotations, video, layer))

    # remove_first_n_frames(path.join(annotations, "amsterdamse_brug/00"), 6)
    # remove_first_n_frames(path.join(annotations, "amsterdamse_brug/01"), 6)