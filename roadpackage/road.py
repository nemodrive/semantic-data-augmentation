import cv2
import os
import pandas as pd
import math
import random
from random import randrange
import numpy as np

ALPHA = 1.29
BETA = -0.61
PERSON_HEIGHT = 150
ELLIPSE_HEIGHT = 20
SEGMENTED_ROAD_COLOR = 0
SEGMENTED_PERSON_COLOR = [0, 0, 0, 255]


def overlay_people_on_road(person_path: str, road: np.ndarray, segmented_road: np.ndarray) \
        -> np.ndarray:

    """
    :param person_path: one column file with all people paths
    :param road: a np.ndarray image with the original road image
    :param segmented_road: a np.ndarray image with the segmented road image
    :return: a np.ndarray image with the overlaid road with people
    """

    # get a random cropped person from PERSON_PATH
    df = pd.read_csv(person_path, sep='\n', header=None)
    idxp = random.randint(0, len(df[0]) - 1)

    # read people image (s_img = people_image)
    s_img = cv2.imread(df[0][idxp], -1)
    s_img = cv2.cvtColor(s_img, cv2.COLOR_BGRA2RGBA)

    # copy road image (l_img = road_image)
    l_img = road

    # copy segmented image of the road(segmented_img = segmented_road_image)
    segmented_img = segmented_road

    # get original road image from ROAD_PATH
    # l_img = cv2.imread(road, -1)

    # get segmented road image from SEGMENTED_ROAD_PATH
    # segmented_img = cv2.imread(segmented_road, -1)

    # get the placement point of person on road
    x2, y2, road_height = find_position_on_road(l_img, segmented_img)

    # if it is not any road in the image, return the original image without augmentation
    if x2 == y2 == -1:
        return l_img

    # set the new coordinates of person in the road image (where he will be placed)
    # y_offset = y_offset - s_img.shape[0]

    # get the new person height using the position on the road
    person_h = person_height(y2, l_img.shape[0], road_height)

    # resize the person image using the new person height (person_h)
    s_img = resize_person(s_img, person_h)

    # calculate the coordinates of the person
    # y1 = y2 - s_img.shape[0]
    # x1 = x2 - s_img.shape[1]

    # person_height = int(linear_function(ALPHA, y2, BETA))

    # resize person image and return image and top-left coordinates

    # draw ellipse of person position
    # draw_person_ellipse_position(l_img, y2)

    # place person on the ellipse
    center = l_img.shape[1] / 2
    y2 = int(y_coordinate_ellipse(x2, center, y2 - ELLIPSE_HEIGHT, center, ELLIPSE_HEIGHT))
    x1 = x2 - s_img.shape[1]
    y1 = y2 - s_img.shape[0]

    # place person on road image
    # alpha_s = s_img[:, :, 3] / 255.0
    # alpha_l = 1.0 - alpha_s

    # overlay person on the road
    overlay_transparent(l_img, s_img, x1, y1)

    # Create the segmented image of person to overlay over segmented road
    # segmented_person = get_segmented_image_of_person(s_img)

    # Show the overlaid image
    # cv2.imshow('image', l_img)
    # cv2.waitKey(0)

    # Show the segmented image
    # cv2.imshow('segmented image', segmented_img)
    # cv2.waitKey(0)

    # Destroy all windows
    # cv2.destroyAllWindows()

    # optional - change color from BGR to RGB (is used in CityscapesSegmentation)
    l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB)

    return l_img


def resize_person(person_image: np.ndarray, height: int)-> np.ndarray:

    # resize the person image, having height equal to PERSON_HEIGHT
    new_height = height
    new_width = int(person_image.shape[1] * new_height / person_image.shape[0])

    # recalculate top-left coordinates
    # x1 = x1 + person_image.shape[1] - new_width
    # y1 = y1 + person_image.shape[0] - new_height

    if new_height == 0 or new_width == 0:
        return person_image

    # resize person_image
    person_image = cv2.resize(person_image, (new_width, new_height))

    return person_image


def person_height(person_pos: int, image_height: int, road_height)-> int:
    # return int(road_height + person_pos - image_height)
    person_percent = ALPHA * person_pos / image_height + BETA
    return int(person_percent * image_height)


def linear_function(alpha: int, x: float, beta: int) -> float:
    return alpha * x + beta


def y_coordinate_ellipse(x: int, x0: int, y0: int, a: int, b: int) -> float:
    return y0 + b * math.sqrt(1 + ((x - x0) / a) ** 2)


def draw_person_ellipse_position(image: np.ndarray, y: int) -> None:
    # draw ellipse over l_img
    for x in range(image.shape[1]):
        x0 = image.shape[1] / 2
        cv2.circle(image,
                   (x, int(y_coordinate_ellipse(x, x0, y - ELLIPSE_HEIGHT, x0, ELLIPSE_HEIGHT))),
                   1, (0, 0, 255), -1)


def find_position_on_road(image, segmented_img: np.ndarray) -> (int, int, int):
    road_points = []
    road_height = 0

    for row in range(segmented_img.shape[0]):
        for column in range(segmented_img.shape[1]):
            if segmented_img[row][column] == SEGMENTED_ROAD_COLOR:
                road_points.append([row, column])
                road_height = max(road_height, segmented_img.shape[0] - row)
                # Draw road
                # cv2.circle(image,
                #            (column, row),
                #            1, (0, 0, 255), -1)

    if len(road_points) == 0:
        return -1, -1, 0

    idr = random.randrange(0, len(road_points))

    return road_points[idr][1], road_points[idr][0], road_height


def get_segmented_image_of_person(image: np.ndarray) -> np.ndarray:
    segmented_person = image
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            if np.all(image[row, column] != [0, 0, 0, 0]):
                segmented_person[row, column] = SEGMENTED_PERSON_COLOR
    return segmented_person


def overlay_transparent(background: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if x < 0:
        w = w + x
        x = x * (-1)
        overlay = overlay[:, x:]
        x = 0

    if y < 0:
        h = h + y
        y = y * (-1)
        overlay = overlay[y:, :]
        y = 0

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background


if __name__ == '__main__':
    road_i = cv2.imread('dataset/cityscapes/out_images/18nov_8c00f08fd4914269-0_88-frame.png', -1)
    seg_road_i = cv2.imread('dataset/cityscapes/out_images/18nov_8c00f08fd4914269-0_trainIds_88.png'
                            , -1)
    overlay_people_on_road('people_path.txt', road_i, seg_road_i)
