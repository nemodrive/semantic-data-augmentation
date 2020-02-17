import os
from matplotlib.path import Path
import matplotlib.pyplot as plt
import json
import numpy as np
import torch
import torchvision


class CSSemanticExtractor:
    def __init__(self, root, bbox_area_threshold=5.0, target_class='person'):
        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if bbox_area_threshold < 0.0:
            bbox_area_threshold = 0
        if bbox_area_threshold > 100.0:
            bbox_area_threshold = 0
        self.bbox_area_threshold = bbox_area_threshold

        self.target_class = target_class
        self.stats = dict()
        self.stats['datasetMode'] = None
        self.stats['datasetSplit'] = None
        self.stats[self.target_class] = dict()
        self.stats[self.target_class]['instancesCount'] = 0
        self.stats[self.target_class]['maxBBox'] = [[0, 0], [0, 0]]
        self.stats[self.target_class]['maxBBoxArea'] = -np.inf
        self.stats[self.target_class]['maxBBoxHeight'] = 0
        self.stats[self.target_class]['maxBBoxWidth'] = 0
        self.stats[self.target_class]['minBBox'] = [[np.inf, np.inf], [np.inf, np.inf]]
        self.stats[self.target_class]['minBBoxArea'] = np.inf
        self.stats[self.target_class]['minBBoxHeight'] = 0
        self.stats[self.target_class]['minBBoxWidth'] = 0

    def __call__(self, dataset, all_instances=False, show_progress=False):
        # Initialize progress counter and stats
        total_progress_count = float(len(dataset))
        progress_count = 0.0

        self.stats['datasetMode'] = dataset.mode
        self.stats['datasetSplit'] = dataset.split

        # Create the necessary folder structure
        target_class_dir = self.root + '/' + self.target_class
        if not os.path.exists(target_class_dir):
            os.makedirs(target_class_dir)

        image_save_idx = 0

        # Step through each image, target and extract instances
        for idx, (image, target) in enumerate(dataset):
            bbox_area_threshold = int(np.round(
                image.shape[1] * image.shape[2] * float(self.bbox_area_threshold) / 100.0
            ))
            for t in target[1]['objects']:
                if t['label'] == self.target_class:
                    bbox = get_bounding_box(t['polygon'])
                    area = bbox_area(bbox)

                    if all_instances or (area >= bbox_area_threshold):
                        # Create destination image
                        # TODO: Add configurable numbering scheme
                        target_image_path = target_class_dir + '/%.6d' % image_save_idx + '.png'
                        target_annotation_path = target_class_dir + '/%.6d' % image_save_idx + '.json'
                        image_save_idx += 1

                        instance = {
                            'source': dataset.obj_wrapper.images[idx],
                            'destination': target_image_path,
                            'imageHeight': int((bbox[1] - bbox[0])[1]),
                            'imageWidth': int((bbox[1] - bbox[0])[0]),
                            'label': t['label'],
                            'bbox': bbox.tolist(),
                            'bboxArea': int(area),
                            'polygon': t['polygon']
                        }

                        cropped_image = semantic_image_crop(image, bbox,
                                                            instance['polygon'],
                                                            instance['imageHeight'],
                                                            instance['imageWidth'])

                        torchvision.utils.save_image(cropped_image, target_image_path)

                        with open(target_annotation_path, 'w') as f:
                            json.dump(instance, f, indent=4)

                        self.__update_stats(instance)

            if show_progress:
                progress_count += 1
                print('Progress: %.2f%% \r' % (progress_count / total_progress_count * 100), end="")

        with open(self.root + '/stats.json', 'w') as f:
            json.dump(self.stats, f, indent=4)

        return self.stats

    def __update_stats(self, instance):
        self.stats[self.target_class]['instancesCount'] += 1
        if instance['bboxArea'] > self.stats[self.target_class]['maxBBoxArea']:
            self.stats[self.target_class]['maxBBoxArea'] = instance['bboxArea']
            self.stats[self.target_class]['maxBBox'] = instance['bbox']
            self.stats[self.target_class]['maxBBoxHeight'] = instance['imageHeight']
            self.stats[self.target_class]['maxBBoxWidth'] = instance['imageWidth']
        if instance['bboxArea'] < self.stats[self.target_class]['minBBoxArea']:
            self.stats[self.target_class]['minBBoxArea'] = instance['bboxArea']
            self.stats[self.target_class]['minBBox'] = instance['bbox']
            self.stats[self.target_class]['minBBoxHeight'] = instance['imageHeight']
            self.stats[self.target_class]['minBBoxWidth'] = instance['imageWidth']


def semantic_image_crop(image, bbox, polygon, height, width):
    x, y = np.meshgrid(np.arange(bbox[0][0], bbox[1][0]), np.arange(bbox[0][1], bbox[1][1]))
    bbox_points = np.vstack((x.flatten(), y.flatten())).T
    polygon_grid = Path(polygon).contains_points(bbox_points)
    polygon_indices = np.argwhere(True == polygon_grid).flatten()
    polygon_points = bbox_points[polygon_indices]

    cropped_image = 255 * torch.ones([image.shape[0], height, width])

    for point in polygon_points:
        for ch in range(image.shape[0]):
            cropped_image[ch][point[1] - bbox[0][1] - 1][point[0] - bbox[0][0] - 1] = \
                image[ch][point[1]][point[0]]

    # TODO: Remove this after debugging
    # fig = plt.figure(figsize=[20, 5])
    # fig.add_subplot(1, 3, 1)
    # plt.imshow(image.permute(1, 2, 0))
    # fig.add_subplot(1, 3, 2)
    # plt.imshow(cropped_image.permute(1, 2, 0))
    # fig.add_subplot(1, 3, 3)
    # plt.imshow(np.asanyarray(target[0]))
    # plt.tight_layout()
    # plt.show(block=True)
    return cropped_image


def get_bounding_box(polygon):
    x, y = zip(*polygon)
    return np.array([[min(x), min(y)], [max(x), max(y)]])


def bbox_area(bbox):
    return np.prod((bbox[1] - bbox[0]))
