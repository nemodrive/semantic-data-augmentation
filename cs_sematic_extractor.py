import numpy
from matplotlib.path import Path

class CSSemanticExtractor:
    def __init__(self, bbox_threshold, target_class):
        self.bbox_threshold = bbox_threshold
        self.target_class = target_class

    def __call__(self, image, target):
        return

    # TODO: this needs to be private
    def get_target_count(self, target, all = False):
        count = 0
        for t in target[1]['objects']:

            # TODO:
            if t['label'] == self.target_class: #and \
                     #(all or (self.get_bounding_box(t['polygon']) == self.bbox_threshold)):
                count += 1
                print(self.get_bounding_box(t['polygon']))
        return count

    # TODO: this needs to be private
    def get_bounding_box(self, polygon):
        x , y = zip(*polygon)
        return [[min(x), min(y)], [max(x), max(y)]]

