import cv2
import numpy as np
import os
import glob

PEOPLE_FOLDER = 'people'
COLOR_DETECTION = np.array([60, 20, 220])
MINIMUM_SIZE_MAN = 4500
PATH_TO_SEGMENTED_CITIES = "gtFine_trainvaltest/gtFine/train/*"
PATH_TO_ORIGINAL_IMAGES = "leftImg8bit/train/"
PATH_TO_PEOPLE_IMAGES = "people/*.png"


def crop_all_people() -> None:
    for cities in glob.glob(PATH_TO_SEGMENTED_CITIES):
        path_to_original_images = PATH_TO_ORIGINAL_IMAGES + cities.split('/')[3]
        path_to_segmented_images = cities + "/*color.png"

        for segmented_image in glob.glob(path_to_segmented_images):
            path_original_image = path_to_original_images + '/*' + \
                                  segmented_image.split('_')[2] + '_' +\
                                  segmented_image.split('_')[3] + \
                                  '*.png'
            original_image = glob.glob(path_original_image)
            get_people_from_cityscape(segmented_image, original_image[0])


def get_people_from_cityscape(segmented_image: str, original_image: str) -> np.ndarray:
    """
    :param segmented_image: ipsum lorem
    :param original_image:
    :return:
    """

    # Read segmented image
    img = cv2.imread(segmented_image)

    # Original image
    img_original = cv2.imread(original_image)

    # Find all people with color equal to COLOR_DETECTION -> in segmented image is [60, 20, 220]
    mask = cv2.inRange(img, COLOR_DETECTION, COLOR_DETECTION)

    # Find contours and hierarchy of all red colored people
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    for i, c in enumerate(contours):

        # Get a rectangle of contoured person
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        box = cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), 2)

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

            # Apply mask to original imagee
            res = cv2.bitwise_and(img_original, img_original, mask=mask_aux)

            # Get cropped image of person
            res = res[y: y + h, x: x + w]

            # Remove background and make it transparent
            res = cv2.cvtColor(res, cv2.COLOR_BGR2BGRA)
            res[np.all(res == [0, 0, 0, 255], axis=2)] = [0, 0, 0, 0]

            # Show cropped man
            # cv2.imshow("Show Boxes", res)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Create new PNG file with the person cropped
            image_name = "person" + str(i) + "_" + \
                         segmented_image.split('_')[1].split('/')[3] + "_" + \
                         segmented_image.split('_')[3].replace("/", "-") + \
                         ".png"
            print(image_name)
            cv2.imwrite(os.path.join(PEOPLE_FOLDER, image_name), res)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('images',
                        help='Two column file with image paths and segmentation paths')
    parser.add_argument('outdir',
                        help='Two column file with image paths and segmentation paths')
    args = parser.parse_args()
