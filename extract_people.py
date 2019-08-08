import cv2
import numpy as np
import os
import argparse
import pandas as pd

PEOPLE_FOLDER = 'people'
COLOR_DETECTION = np.array([60, 20, 220])
MINIMUM_SIZE_MAN = 4500
PATH_TO_SEGMENTED_CITIES = "dataset/gtFine_trainvaltest/gtFine/train/*"
PATH_TO_ORIGINAL_IMAGES = "dataset/leftImg8bit/train/"
PATH_TO_PEOPLE_IMAGES = "people/*.png"
SEGMENTED_IMAGE_EXTRA_PATH = 'dataset/gtFine_trainvaltest/'
ORIGINAL_IMAGE_EXTRA_PATH = 'dataset/'
counter_files = 0


def crop_all_people(people_link_file: str) -> None:
    """
    :param people_link_file: a two column file with original cityscape image path and segmented
                            cityscape image path
    :return:
    """

    file_path_people = open("people_path.txt", "w")
    file_path_people.write("")
    file_path_people.close()

    df = pd.read_csv(people_link_file, sep='\t', header=None)
    df[0] = ORIGINAL_IMAGE_EXTRA_PATH + df[0]
    df[1] = SEGMENTED_IMAGE_EXTRA_PATH + df[1]

    for index in range(len(df)):
        extract_people_from_cityscape(df[1][index], df[0][index])

    # for cities in glob.glob(PATH_TO_SEGMENTED_CITIES):
    #     path_to_original_images = PATH_TO_ORIGINAL_IMAGES + cities.split('/')[4]
    #     path_to_segmented_images = cities + "/*color.png"
    #
    #     for segmented_image in glob.glob(path_to_segmented_images):
    #         path_original_image = path_to_original_images + '/*' + \
    #                               segmented_image.split('_')[2] + '_' +\
    #                               segmented_image.split('_')[3] + \
    #                               '*.png'
    #         original_image = glob.glob(path_original_image)
    #         extract_people_from_cityscape(segmented_image, original_image[0])


def extract_people_from_cityscape(segmented_image: str, original_image: str)-> None:
    """
    :param segmented_image: segmented image path
    :param original_image: original image path
    :return:
    """

    # used to counter images and write in file name
    global counter_files

    # Read segmented image
    img = cv2.imread(segmented_image)

    # Original image
    img_original = cv2.imread(original_image)

    # Find all people with color equal to COLOR_DETECTION -> in segmented image is [60, 20, 220]
    mask = cv2.inRange(img, COLOR_DETECTION, COLOR_DETECTION)

    # Find contours and hierarchy of all red colored people
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    for i, c in enumerate(contours):

        # Get a rectangle of contoured person
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        # box = cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Get a cropped person including other people in background
        cropped = mask[y: y+h, x: x+w]

        # Copy mask -> we will remove other people from background on this mask
        mask_aux = mask.copy()

        # Check if the person has size greater than MINIMUM_SIZE_MAN
        if cropped.shape[0] * cropped.shape[1] > MINIMUM_SIZE_MAN:

            # Remove other people from background
            for i2, c2 in enumerate(contours):
                rect2 = cv2.boundingRect(c2)
                if hierarchy[0, i2, 3] != i and i != i2:
                    mask_aux[rect2[1]: rect2[1] + rect2[3], rect2[0]: rect2[0] + rect2[2]] = [0]

            # Apply mask to original image
            res = cv2.bitwise_and(img_original, img_original, mask=mask_aux)

            # Get cropped image of person
            res = res[y: y + h, x: x + w]

            # Remove background and make it transparent
            res = cv2.cvtColor(res, cv2.COLOR_BGR2BGRA)
            res[np.all(res == [0, 0, 0, 255], axis=2)] = [0, 0, 0, 0]

            # Create new PNG file with the person cropped
            image_name = "person" + str(i) + "_" + \
                         segmented_image.split('_')[1].split('/')[3] + "_" + \
                         str(counter_files) + \
                         ".png"
            counter_files += 1
            print(image_name)
            cv2.imwrite(os.path.join(PEOPLE_FOLDER, image_name), res)

            file_path_people = open("people_path.txt", "a")
            file_path_people.write("people/" + image_name + "\n")
            file_path_people.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract people from cityscape')
    parser.add_argument('link_file',
                        help='Two column file with image paths and segmentation paths')
    args = parser.parse_args()
    link_file = args.link_file

    if not os.path.isfile(link_file):
        print("No file with name " + link_file + " found.")

    crop_all_people(link_file)
