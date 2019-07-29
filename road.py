import cv2
import math
from random import randrange
import numpy as np

ALPHA = 1.35
BETA = -250
PERSON_HEIGHT = 150
ELLIPSE_HEIGHT = 20
SEGMENTED_ROAD_COLOR = [128, 64, 128]
SEGMENTED_PERSON_COLOR = [255, 0, 0, 255]


def overlay_people_on_road(person_path: str, road_path, segmented_road_path) -> None:

    # get cropped person from PERSON_PATH
    s_img = cv2.imread(person_path, -1)

    # get original road image from ROAD_PATH
    l_img = cv2.imread(road_path, -1)

    # get segmented road image from SEGMENTED_ROAD_PATH
    segmented_img = cv2.imread(segmented_road_path, -1)

    # get the placement point of person on road
    x_offset, y_offset = find_position_on_road(segmented_img, s_img.shape[1], s_img.shape[0])

    # set the new coordinates of person in the road image (where he will be placed)
    y_offset = y_offset - s_img.shape[0]

    # calculate the coordinates of the person
    y1, y2 = y_offset, y_offset + s_img.shape[0]
    x1, x2 = x_offset, x_offset + s_img.shape[1]

    person_height = int(linear_function(ALPHA, y2, BETA))

    # resize person image and return image and top-left coordinates
    s_img, x1, y1 = resize_person(s_img, person_height, x1, y1)

    # draw ellipse of person position
    # draw_person_ellipse_position(l_img, y2)

    # place person on the ellipse
    center = l_img.shape[1] / 2
    y2 = int(y_coordinate_ellipse(x2, center, y2 - ELLIPSE_HEIGHT, center, ELLIPSE_HEIGHT))
    x1 = x2 - s_img.shape[1]
    y1 = y2 - s_img.shape[0]

    # place person on road image
    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    print(x1, y1)
    overlay_transparent(l_img, s_img, x1, y1)

    # Create the segmented image of person to overlay over segmented road
    # segmented_person = get_segmented_image_of_person(s_img)

    # for c in range(0, 3):
    #     segmented_img[y1:y2, x1:x2, c] = (alpha_s * segmented_person[:, :, c] + alpha_l * segmented_img[y1:y2, x1:x2, c])

    # Show the overlaid image
    cv2.imshow('image', l_img)
    cv2.waitKey(0)

    # Show the segmented image
    # cv2.imshow('segmented image', segmented_img)
    # cv2.waitKey(0)

    # Destroy all windows
    cv2.destroyAllWindows()


def resize_person(person_image, height: int, x1: int, y1: int):

    # resize the person image, having height equal to PERSON_HEIGHT
    new_height = height
    new_width = int(person_image.shape[1] * new_height / person_image.shape[0])

    # recalculate top-left coordinates
    x1 = x1 + person_image.shape[1] - new_width
    y1 = y1 + person_image.shape[0] - new_height

    # resize person_image
    person_image = cv2.resize(person_image, (new_width, new_height))

    return person_image, x1, y1


def linear_function(alpha: int, x: int, beta: int) -> int:
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


def find_position_on_road(segmented_img, person_width, person_height):
    while True:
        y_pos = randrange(0, segmented_img.shape[0])
        x_pos = randrange(0, segmented_img.shape[1] - person_width - 1)

        left = segmented_img[y_pos, x_pos]
        mid = segmented_img[y_pos, x_pos + int(person_width / 2)]
        right = segmented_img[y_pos, x_pos + person_width]

        if np.all(left == SEGMENTED_ROAD_COLOR) and \
                np.all(right == SEGMENTED_ROAD_COLOR) and \
                np.all(mid == SEGMENTED_ROAD_COLOR):
            return x_pos, y_pos


def get_segmented_image_of_person(image):
    segmented_person = image
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            if np.all(image[row, column] != [0, 0, 0, 0]):
                segmented_person[row, column] = SEGMENTED_PERSON_COLOR
    return segmented_person


def overlay_transparent(background, overlay, x, y):

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

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background
