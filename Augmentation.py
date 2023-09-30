import math
import skimage
import pathlib
import numpy as np
from skimage.exposure import is_low_contrast, equalize_adapthist

IN = pathlib.WindowsPath("C:\\Users\\fahad\\Downloads\\Whatever\\Data\\train\\normal")
# Define the necessary operations for data augmentation
ROTATE_90 = 1
ROTATE_180 = 2
ROTATE_270 = 3
FLIP_H = 4
FLIP_V = 5

def crop_and_contrast(input_dir, targetsize=(256, 256)):
    input_dir = input_dir.absolute()
    if input_dir.name.endswith('.png') or input_dir.name.endswith('.jpg') or input_dir.name.endswith('.jpeg'):
        # load the PNG or JPEG image here as a grayscale
        image = skimage.io.imread(input_dir, as_gray=True)
        width, height = image.shape[:2]
        # check if the image is already in the appropriate size, e.g. 256x256
        if (width == targetsize[0] or height == targetsize[1]) and math.fabs(width - height) < 20:
            pass
        else:
            aspect_ratio = float(width) / float(height)
            if aspect_ratio > 1:
                new_width = targetsize[0]
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = targetsize[1]
                new_width = int(new_height * aspect_ratio)
            image = skimage.transform.resize(image, output_shape=(new_width, new_height), preserve_range=True, anti_aliasing=False)

        # increase the contrast of the image
        if is_low_contrast(image):
            image = equalize_adapthist(image, kernel_size=3, clip_limit=0.35)

        # initialize a random integer for data augmentation
        skimage.io.imsave(input_dir, image)

def appending(input_dir, add_string):
    input_dir = input_dir.absolute()
    if input_dir.is_file():
        if add_string in input_dir.stem.lower():
            return
        else:
            new_name = input_dir.with_stem(input_dir.stem + add_string)
            return new_name


def augmentation(input_dir, maximum):
    input_dir = input_dir.absolute()
    count = 0
    subs = []
    for sub in input_dir.iterdir():
        if sub.is_dir():
            pass
        elif sub.is_file():
            sub = sub.absolute()
            if sub.name.endswith('.png') or sub.name.endswith('.jpg') or sub.name.endswith('jpeg'):
                count = count + 1
                subs.append(sub)
    if count >= maximum:
        return
    else:
        while count < maximum:
            for subfile in subs:
                lower = subfile.stem.lower()
                if "rotated90" in lower \
                        or "rotated180" in lower \
                        or "rotated270" in lower \
                        or "flippedh" in lower   \
                        or "flippedv" in lower:
                    continue
                subfile = subfile.absolute()
                image = skimage.io.imread(subfile, as_gray=True)
                # choose a random preprocessing operation out of those declared in the very start
                operation = np.random.randint(1, 6)
                new_image_path = pathlib.Path(subfile)
                if operation == ROTATE_90:
                    image = skimage.transform.rotate(image, 90)
                    new_image_path = appending(subfile, "rotated90")
                elif operation == ROTATE_180:
                    image = skimage.transform.rotate(image, 180)
                    new_image_path = appending(subfile, "rotated180")
                elif operation == ROTATE_270:
                    image = skimage.transform.rotate(image, 270)
                    new_image_path = appending(subfile, "rotated270")
                elif operation == FLIP_H:
                    image = np.flip(image, 1)
                    new_image_path = appending(subfile, "flippedH")
                elif operation == FLIP_V:
                    image = np.flip(image, 0)
                    new_image_path = appending(subfile, "flippedV")
                skimage.io.imsave(new_image_path, image)
                count = count + 1
                subs.append(new_image_path)
                if count >= maximum:
                    return

def main(input_directory):
    input_directory = input_directory.absolute()
    for sub in input_directory.iterdir():
        sub = sub.absolute()
        if sub.is_dir():
            main(sub)
        elif sub.is_file():
            crop_and_contrast(sub)


augmentation(IN, 465)

