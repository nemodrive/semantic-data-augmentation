import os
from person import crop_all_people
from road import overlay_people_on_road
# import imageio
# import imgaug as ia
# from imgaug import augmenters as iaa
import glob
import random
import pandas as pd


PERSON_PATH = 'people/person1_monchengladbach_019500.png'
ROAD_PATH = 'dataset/nemodrive/cityscapes/out_images/18nov_8c00f08fd4914269-0_55-frame.png'
SEGMENTED_ROAD_PATH = 'dataset/nemodrive/cityscapes/out_images/18nov_8c00f08fd4914269-0_55.png'


# def augment_image_test():
#     image = imageio.imread("https://upload.wikimedia.org/wikipedia/en/7/7d/" + \
#                            "Lenna_%28test_image%29.png")
#     rotate = iaa.Affine(rotate=(-25, 25))
#     image_aug = rotate.augment_image(image)
#     ia.imshow(image_aug)


if __name__ == '__main__':
    # overlay_people_on_road(PERSON_PATH, ROAD_PATH, SEGMENTED_ROAD_PATH)
    # crop_all_people()

    data_paths_file = "dataset/cityscapes/train_fine.txt"
    df = pd.read_csv(data_paths_file, sep='\t', header=None)
    df[0] = "dataset/cityscapes/" + df[0]
    df[1] = "dataset/cityscapes/" + df[1]
    df[2] = df[1].apply(lambda x: x.replace("_trainIds", ""))

    people = glob.glob("people/*.png")

    while True:
        idxp = random.randint(0, len(people))
        idxr = random.randint(0, len(df))
        print(people[idxp], df.loc[idxr][0], df.loc[idxr][2])
        overlay_people_on_road(people[idxp], df.loc[idxr][0], df.loc[idxr][2])



