#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Superclass of weight

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""

from __future__ import print_function, division, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import operator
from . import logger


class WeightBase(object):
    """
    The superclass for weighting. All the weighting results is supposed
    to be kept in the point.weight.
    """

    def __init__(self, points, sort_by_tag=False, remove_duplicate=False):
        """
        :param points: list of points
        :type points: list
        :param sort_by_tag: sort the points by tag. This will change the
            order of points in origin points list. We usually set the tag
            as "network.station". So this sort will group stations in the
            same network together. If you want to refer to the origin list
            of points, use self._points
        :type sort_by_tag: bool
        :param remove_duplicate: remove the points with the same coordinate.
            This flag should be set to true to certain methods, for example,
            sphere voronoi because two points at the same location will
            cause problem in sphere area partition.
        :type remove_duplicate: bool

        """

        self.points = points
        self._points = points[:]

        self._check_points_dims()
        self.detect_duplicate_tags()
        if sort_by_tag:
            self._sort_points_by_tag()

        self.remove_duplicate = remove_duplicate
        self.removed_points = []
        if remove_duplicate:
            self._remove_duplicate_points()

    def _check_points_dims(self):
        dims = [point.coordinate.shape for point in self.points]
        if len(set(dims)) != 1:
            raise ValueError("Points are with different dimensions: %s"
                             % set(dims))

    @property
    def points_dimension(self):
        return self.points[0].coordinate.shape

    @property
    def npoints(self):
        return len(self.points)

    @property
    def points_tags(self):
        return [point.tag for point in self.points]

    @property
    def condition_number(self):
        """
        Condition number is defined as max weight over min weight
        """
        weight = self.points_weights
        if np.min(weight) == 0:
            print("The min value of weights is 0. Condition number not"
                  "achievable")
            return None
        else:
            return np.max(weight) / np.min(weight)

    @property
    def points_coordinates(self):
        return np.array([point.coordinate for point in self.points])

    @property
    def points_weights(self):
        return np.array([point.weight for point in self.points])

    @points_weights.setter
    def points_weights(self, weights):
        if len(weights) != self.npoints:
            raise ValueError("Dimension of weight(%d) not same as npoints(%d)"
                             % (len(weights), self.npoints))
        for idx, point in enumerate(self.points):
            _w = weights[idx]
            if _w < 0:
                raise ValueError("weight[%d] is negative: %f!" % (idx, _w))
            point.weight = _w

    def detect_duplicate_tags(self):
        tag_list = self.points_tags
        tag_set = set(tag_list)
        if len(tag_set) != len(tag_list):
            for tag in tag_set:
                tag_list.remove(tag)
            raise ValueError("Duplicate tags detected: %s" % tag_list)

    def _sort_points_by_tag(self):
        """
        Sort station by its tag
        """
        self.points = sorted(self.points, key=lambda x: x.tag)

    def _find_duplicate_coordinates(self):
        """
        Find duplicate points coordinates
        """
        def _unique_row(array):
            _arr = np.ascontiguousarray(array)
            _arr2 = _arr.view(np.dtype(
                (np.void, _arr.dtype.itemsize * _arr.shape[1])))
            _, idx = np.unique(_arr2, return_index=True)
            return idx

        # check coordinates
        coords = self.points_coordinates
        unique_idxs = _unique_row(coords)

        npts = self.npoints
        dup_idxs = list(set(range(npts)) - set(unique_idxs))
        if len(unique_idxs) + len(dup_idxs) != npts:
            raise ValueError("Find duplicate coordinates failed!!! Error!!!"
                             "Please report to the developer")
        dup_idxs.sort()
        return dup_idxs

    def _remove_duplicate_points(self):
        """
        Remove the points with duplicated coordinates. This step is
        required by some weighting strategy like spherevoronoi.
        :return:
        """

        dup_idxs = self._find_duplicate_coordinates()

        if len(dup_idxs) == 0:
            logger.info("No duplicate coordinates found")
            return

        for index in sorted(dup_idxs, reverse=True):
            self.removed_points.append(self.points[index])
            del self.points[index]

        if len(self._find_duplicate_coordinates()) != 0:
            raise ValueError("There are still duplicates after removing.")

        logger.info("Number of original npoints: %d" % len(self._points))
        logger.info("Number of points removed(duplicate coordinates): %d" %
                    len(dup_idxs))
        for _point in self.removed_points:
            logger.debug("Points removed: %s" % _point)
        logger.info("Number of remaining stations: %d" % self.npoints)

    def normalize_weight(self, mode="average"):
        """
        Normalize the weights while keep the ratio of points weights the
        same. Options are:
            1) "average": normalize the average of weights to 1
            2) "max": normalize the max of weights to 1
            3) "sum": normalize the sum of weights to 1
        :return:
        """
        array = self.points_weights
        if mode == "average":
            # average to 1
            norm_factor = (len(array) / sum(array))
        elif mode == "max":
            # max to 1
            norm_factor = 1.0 / max(array)
        elif mode == "sum":
            # sum to 1
            norm_factor = 1.0 / sum(array)

        array = array * norm_factor
        self.points_weights = array

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
