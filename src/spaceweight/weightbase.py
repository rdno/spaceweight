#!/usr/bin/env python
"""
Contains several distance and azimuth weighting strategy on
# surface of a sphere
# Distance: 1) Relative Distance Weighting
#           2) Voronoi Weighting(a slighting modified version of:
#              https://github.com/tylerjereddy/py_sphere_Voronoi)
#           3) Exponential Distance Weighting(from CMT3D, by Qinya Liu)
# Azimuth:  1) Relative Azimuth Weighting
#           2) Bin Azimuth Weighting(from CMT3D, by Qinya Liu)
"""

from __future__ import print_function, division, absolute_import
import copy
import numpy as np
import matplotlib.pyplot as plt
import collections
import operator
from . import logger


class WeightBase(object):

    def __init__(self, points, sort_by_tag=False, remove_duplicate=False,
                 normalize_flag=True, normalize_mode="average"):

        self.points = points
        self._points = copy.deepcopy(points)

        self.dimension = self._check_points_dims()

        if sort_by_tag:
            self._sort_points_by_tag()

        self.remove_duplicate = remove_duplicate
        if remove_duplicate:
            self._remove_duplicate_points()

        self.points_coordinates = self._get_points_coordinates()

        normalize_mode = normalize_mode.lower()
        norm_modes = ["max", "sum", "average"]
        if normalize_flag and normalize_mode not in norm_modes:
            raise ValueError("Normalize_mode(%s) not in current options(%s)"
                             % (normalize_mode, norm_modes))
        self.normalize_flag = normalize_flag
        self.normalize_mode = normalize_mode

    def _check_points_dims(self):
        default_dim = self.points[0].coordinate.shape
        for point in self.points:
            if point.coordinate.shape != default_dim:
                raise ValueError("Dimension of points coordinate is not the "
                                 "Same!")
        return default_dim[0]

    @property
    def npoints(self):
        return len(self.points)

    @property
    def points_tags(self):
        tags = []
        for point in self.points:
            tags.append(point.tag)
        return tags

    @property
    def condition_number(self):
        weight = self.points_weights
        return np.max(weight) / np.min(weight)

    def _get_points_coordinates(self):
        coords = []
        for point in self.points:
            coords.append(point.coordinate)
        return np.array(coords)

    @property
    def points_weights(self):
        weights = []
        for point in self.points:
            weights.append(point.weight)
        return np.array(weights)

    def _set_points_weights(self, weight):
        if len(weight) != self.npoints:
            raise ValueError("Dimension of weight(%d) not same as npoints(%d)"
                             % (len(weight), self.npoints))
        for idx, point in enumerate(self.points):
            point.weight = weight[idx]

    def _sort_points_by_tag(self):
        """
        Sort station by its tag
        """
        point_dict = dict()
        point_list = list()
        done_tags = []
        for point in self.points:
            point_dict[point.tag] = point
            if point.tag in done_tags:
                raise ValueError("Tags is duplicated: %s" % point.tag)
            done_tags.append(point.tag)

        od = collections.OrderedDict(sorted(point_dict.items()))
        for key, value in od.items():
            point_list.append(value)
        self.points = point_list

    def _find_duplicate_coordinates(self):
        """
        Find duplicate points(by tag and coordinates)
        """
        # check coordinates
        coords = self._get_points_coordinates()
        dim = coords.shape[0]
        b = np.ascontiguousarray(coords).view(np.dtype(
            (np.void, coords.dtype.itemsize * coords.shape[1])))
        _, idx = np.unique(b, return_index=True)
        duplicate_list = list(set([i for i in range(dim)]) - set(idx))
        if len(idx) + len(duplicate_list) != dim:
            raise ValueError("The sum of dim doesn't agree")
        duplicate_list.sort()
        return duplicate_list

    def _remove_duplicate_points(self):

        duplicate_list = self._find_duplicate_coordinates()

        if len(duplicate_list) == 0:
            logger.info("No duplicate coordinates found")
            return

        removed_points = []
        for index in sorted(duplicate_list, reverse=True):
            removed_points.append(self.points[index])
            del self.points[index]

        if len(self._find_duplicate_coordinates()) != 0:
            raise ValueError("There are still duplicates after removing.")

        logger.info("Number of original npoints: %d" % len(self._points))
        logger.info("Number of points removed(duplicate coordinates): %d" %
                    len(duplicate_list))
        for _point in removed_points:
            logger.debug("Points removed: %s" % _point)
        logger.info("Number of remaining stations: %d" % self.npoints)

    def _normalize_weight(self):
        mode = self.normalize_mode
        array = self.points_weights
        if mode == "average":
            # average to 1
            sum_array = sum(array)
            array *= (len(array) / sum_array)
        elif mode == "max":
            # max to 1
            array /= array.max()
        elif mode == "sum":
            # sum to 1
            sum_array = sum(array)
            array /= sum_array
        self._set_points_weights(array)

    def write_weight(self, filename="weight.txt", order="tag"):

        weight_dict = dict()
        order = order.lower()
        _options = ["tag", "weight"]
        if order not in _options:
            raise ValueError("Order(%s) must be in:%s" % (order,
                                                          _options))

        for point in self.points:
            weight_dict[point.tag] = point.weight
        if order == "tag":
            _sorted = sorted(weight_dict.items(), key=operator.itemgetter(0))
        elif order == "weight":
            _sorted = sorted(weight_dict.items(), key=operator.itemgetter(1))
        else:
            raise NotImplementedError("order options: %s" % _options)

        with open(filename, 'w') as fh:
            for tag, weight in _sorted:
                fh.write("%-10s %15.5e\n" % (tag, weight))

    def plot_weight_histogram(self, figname=None):

        fig = plt.figure(figsize=(6, 5))
        plt.hist(self.points_weights, bins=30, alpha=0.75)
        plt.xlabel("Weight")
        plt.ylabel("Count")
        plt.title("Histrogram of weights")
        plt.grid(True)

        plt.tight_layout()
        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)
            plt.close(fig)
