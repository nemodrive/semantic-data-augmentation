import numpy
from matplotlib.path import Path
import json
import numpy as np


class CSSemanticExtractor:
    def __init__(self, bbox_area_threshold=30000, target_classes=['person']):
        self.bbox_area_threshold = bbox_area_threshold
        self.target_classes = target_classes
        self.stats = dict()

    def __call__(self, dataset, all_instances=False, show_progress=False):

        total_progress_count = float(len(dataset)*len(self.target_classes))
        progress_count = 0.0
        for target_class in self.target_classes:
            self.stats[target_class] = dict()
            self.stats[target_class]['count'] = 0
            for image, target in dataset:
                self.stats[target_class]['count'] += \
                    self._get_target_class_count(target, target_class, all_instances)

                if show_progress:
                    progress_count += 1
                    print('Progress: %.2f%% \r' % (progress_count / total_progress_count * 100), end="")

        return self.stats

    def _get_target_class_count(self, target, target_class, all_instances=False):
        count = 0
        for t in target[1]['objects']:
            bbox_area = self._bbox_area(self.__get_bounding_box(t['polygon']))
            if t['label'] == target_class and \
                    (all_instances or (bbox_area <= self.bbox_area_threshold)):
                count += 1

        return count

    @staticmethod
    def __get_bounding_box(polygon):
        x , y = zip(*polygon)
        return np.array([[min(x), min(y)], [max(x), max(y)]])

    @staticmethod
    def _bbox_area(bbox):
        return np.prod((bbox[1] - bbox[0]))

