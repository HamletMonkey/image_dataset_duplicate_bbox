import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def get_filelist(IMG_PATH):
    file = [f.parts[-1].split(".")[0] for f in Path(IMG_PATH).iterdir()]
    return file


def get_imgsize(IMG_PATH):
    img = Image.open(IMG_PATH)
    img = np.array(img)
    return img.shape


# resize, greyscale image - output image array
def img_array(IMG_PATH, size=(64, 64), grey=False):
    img = Image.open(IMG_PATH)
    img = ImageOps.pad(img, size, color="black")
    if grey:
        img = ImageOps.grayscale(img)
    img_array = np.array(img)
    return img_array


def imgfolder_array(folder, size, grey):
    filename = get_filelist(folder)
    result = []
    for item in filename:
        r = img_array(os.path.join(folder, f"{item}.jpg"), size, grey)
        result.append(r)
    X = np.stack(result)
    print(X.shape)
    return X


def chunk_list(filelist, size):
    for i in range(0, len(filelist), size):
        a = filelist[i : i + size]
        yield a
