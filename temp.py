from re import I
from PIL import Image
from os import path, listdir, remove


def png2jpg(directory):
    for fn in sorted(listdir(directory)):
        name, ext = path.splitext(fn)
        img = Image.open(path.join(directory, fn))
        img = img.convert('RGB')

        img.save(path.join(directory, f"{name}.jpg"))
        if ext == ".png":
            remove(path.join(directory, f"{name}.png"))


if __name__ == "__main__":
    dirs = [
        "thesis/datasets/Videos/Images/amsterdamse_brug",
        "thesis/datasets/Videos/Images/ringdijk",
        "thesis/datasets/Videos/Images/nescio_2",
        "thesis/datasets/Videos/Images/nescio_1",
    ]

    for directory in dirs:
        png2jpg(directory)