import os
from matplotlib.path import Path
import matplotlib.pyplot as plt
import json
import numpy as np


class CSSemanticExtractor:
    def __init__(self, root, bbox_area_threshold=450688, target_classes=['person']):
        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.bbox_area_threshold = bbox_area_threshold
        self.target_classes = target_classes
        self.stats = dict()

    def __call__(self, dataset, all_instances=False, show_progress=False):
        # Initialize progress counter
        total_progress_count = float(len(dataset) * len(self.target_classes))
        progress_count = 0.0

        self.stats['datasetMode'] = dataset.mode
        self.stats['datasetSplit'] = dataset.split
        self.stats['targetClasses'] = self.target_classes

        # Step through each semantic target class we want to extract
        for target_class in self.target_classes:

            # Create the necessary folder structure
            target_class_dir = self.root + '/' + target_class
            if not os.path.exists(target_class_dir):
                os.makedirs(target_class_dir)

            # Initialize extraction stats for this target class
            self.__init_target_class_stats(target_class)
            image_idx = 0

            # Step through each image, target and extract instances
            for idx, (image, target) in enumerate(dataset):
                for t in target[1]['objects']:
                    if t['label'] == target_class:
                        bbox = self.__get_bounding_box(t['polygon'])
                        bbox_area = self.__bbox_area(bbox)

                        if all_instances or (bbox_area >= self.bbox_area_threshold):
                            # Create destination image
                            # TODO: Add numbering check if some images already exist
                            # TODO: Add configurable numbering scheme
                            target_image_path = target_class_dir + '/%.6d' % image_idx + '.png'
                            target_annotation_path = target_class_dir + '/%.6d' % image_idx + '.json'
                            image_idx += 1

                            instance = {
                                'source': dataset.obj_wrapper.images[idx],
                                'destination': target_image_path,
                                'imageHeight': int((bbox[1] - bbox[0])[1]),
                                'imageWidth': int((bbox[1] - bbox[0])[0]),
                                'label': t['label'],
                                'bbox': bbox.tolist(),
                                'polygon': t['polygon']
                            }

                            self.__semantic_image_crop(image, bbox, t['polygon'])

                            with open(target_annotation_path, 'w') as f:
                                json.dump(instance, f, indent=4)

                            self.__update_stats(bbox, bbox_area, target_class)

                if show_progress:
                    progress_count += 1
                    print('Progress: %.2f%% \r' % (progress_count / total_progress_count * 100), end="")

        return self.stats

    def __init_target_class_stats(self, target_class):
        self.stats[target_class] = dict()
        self.stats[target_class]['instancesCount'] = 0
        self.stats[target_class]['maxBBox'] = np.array([[0, 0], [0, 0]])
        self.stats[target_class]['maxBBoxArea'] = -np.inf
        self.stats[target_class]['minBBox'] = np.array([[np.inf, np.inf], [np.inf, np.inf]])
        self.stats[target_class]['minBBoxArea'] = np.inf

    def __update_stats(self, bbox, bbox_area, target_class):
        self.stats[target_class]['instancesCount'] += 1
        if bbox_area > self.stats[target_class]['maxBBoxArea']:
            self.stats[target_class]['maxBBoxArea'] = bbox_area
            self.stats[target_class]['maxBBox'] = bbox.tolist()
        if bbox_area < self.stats[target_class]['minBBoxArea']:
            self.stats[target_class]['minBBoxArea'] = bbox_area
            self.stats[target_class]['minBBox'] = bbox.tolist()

    @staticmethod
    def __semantic_image_crop(image, bbox, polygon):
        polygon = Path(polygon)
        x, y = np.meshgrid(np.arange(bbox[0][0], bbox[1][0]), np.arange(bbox[0][1], bbox[1][1]))
        points = np.vstack((x.flatten(), y.flatten())).T
        grid = polygon.contains_points(points)
        idx = np.argwhere(grid == True).flatten()

        image = image.permute(1, 2, 0)

        # TODO: Compute location original image location
        # TODO: Compute target tensor



        # plt.figure()
        # plt.imshow(image)
        # plt.show()

        # fig = plt.figure(figsize=[20, 5])
        # fig.add_subplot(1, 2, 1)
        # plt.imshow(image.permute(1, 2, 0))
        # # fig.add_subplot(1, 2, 2)
        # # plt.imshow(np.asanyarray(target[0]))
        # plt.tight_layout()
        # plt.show(block=True)
        return

    @staticmethod
    def __get_bounding_box(polygon):
        x, y = zip(*polygon)
        return np.array([[min(x), min(y)], [max(x), max(y)]])

    @staticmethod
    def __bbox_area(bbox):
        return np.prod((bbox[1] - bbox[0]))
