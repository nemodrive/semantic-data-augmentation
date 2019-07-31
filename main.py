import os
from road import overlay_people_on_road
# import imageio
# import imgaug as ia
# from imgaug import augmenters as iaa
import glob
import random
import pandas as pd


CITYSCAPE_PATH = "dataset/cityscapes/"
DATA_PATH_FILE = "dataset/cityscapes/train_fine.txt"


# def augment_image_test():
#     image = imageio.imread("https://upload.wikimedia.org/wikipedia/en/7/7d/" + \
#                            "Lenna_%28test_image%29.png")
#     rotate = iaa.Affine(rotate=(-25, 25))
#     image_aug = rotate.augment_image(image)
#     ia.imshow(image_aug)


def overlay_all_road_images_with_people():
    data_paths_file = DATA_PATH_FILE
    df = pd.read_csv(data_paths_file, sep='\t', header=None)
    df[0] = CITYSCAPE_PATH + df[0]
    df[1] = CITYSCAPE_PATH + df[1]

    people = glob.glob("people/*.png")

    while True:
        idxp = random.randint(0, len(people))
        idxr = random.randint(0, len(df))
        print(people[idxp], df.loc[idxr][0], df.loc[idxr][1])
        overlay_people_on_road(people[idxp], df.loc[idxr][0], df.loc[idxr][1])


if __name__ == '__main__':
    overlay_all_road_images_with_people()






