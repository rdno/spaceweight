#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class that contains several azimuth weighting strategy on
 surface of a sphere

 Azimuth:  1) Relative Azimuth Weighting
           2) Bin Azimuth Weighting(from CMT3D, by Qinya Liu)
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt

from . import logger
from .sphereweightbase import SphereWeightBase
from .plot_util import plot_2d_matrix
from .util import search_for_ratio


class SphereAziBin(SphereWeightBase):
    """
    Sort Azimuth into certain number of bins and determine weight.
    Used in original version of CMT3D(Qinya Liu).
    """

    def __init__(self, points, bin_order=0.5, nbins=12, center=None,
                 sort_by_tag=False, remove_duplicate=False,
                 normalize_mode="average"):
        """
        :param ref_distance: reference epicenter distance(unit: degree)
        """
        SphereWeightBase.__init__(self, points, center=center,
                                  sort_by_tag=sort_by_tag,
                                  remove_duplicate=remove_duplicate,
                                  normalize_mode=normalize_mode)

        self.bin_order = bin_order
        self.nbins = nbins

    def _weight_function(self, azi_count):
        return (1.0 / azi_count) ** self.bin_order

    def calculate_weight(self):
        weight = np.zeros(self.npoints)

        azi_array, azi_bin, azi_bin_dict = \
            self._stats_azimuth_info(self.nbins)

        for bin_idx, sta_list in azi_bin_dict.iteritems():
            count_in_bin = len(sta_list)
            for _sta_idx in sta_list:
                weight[_sta_idx] = self._weight_function(count_in_bin)

        self.points_weights = weight
        self.normalize_weight()

        logger.info("Number of points at this stage: %d" % self.npoints)
        logger.info("Condition number of weight array(max/min): %8.2f"
                    % self.condition_number)


class SphereAziRel(SphereWeightBase):

    def __init__(self, points, center=None,
                 sort_by_tag=False, remove_duplicate=False,
                 normalize_mode="average"):
        SphereWeightBase.__init__(self, points, center=center,
                                  sort_by_tag=sort_by_tag,
                                  remove_duplicate=remove_duplicate,
                                  normalize_mode=normalize_mode)

    def _build_azimuth_matrix(self):

        azi_array = self._calculate_azimuth_array()

        npts = self.npoints
        azi_matrix = np.zeros([npts, npts])
        for _i in range(npts):
            for _j in range(_i+1, npts):
                azi_matrix[_i, _j] = azi_array[_i] - azi_array[_j]
                azi_matrix[_j, _i] = - azi_matrix[_i, _j]

        np.fill_diagonal(azi_matrix, 0.0)

        return azi_matrix

    @staticmethod
    def _transfer_azi_to_weight(azi_m, ref_azi):
        """
        Transfer the azimuth matrix into weight matrix by a given
        reference azimuth(distance unit is degree)

        :param azi_m: azimuth matrix
        :param ref_azi: reference azimuth
        :return:
        """
        exp_matrix = np.exp(-(azi_m / ref_azi)**2)
        sum_exp = np.sum(exp_matrix, axis=1)
        weight = 1. / sum_exp
        return weight, exp_matrix

    def calculate_weight(self, ref_azimuth):

        azi_matrix = self._build_azimuth_matrix()

        weight, self.exp_matrix = \
            self._transfer_azi_to_weight(azi_matrix, ref_azimuth)

        self.points_weights = weight
        self.normalize_weight()

        logger.info("Number of points at this stage: %d" % self.npoints)
        logger.info("Condition number of weight array(max/min): %8.2f"
                    % self.condition_number)

    def scan(self, start=0.5, end=10.0, gap=0.5, plot=False, figname=None):
        """
        Scan among the range of ref_dists and return the condition number
        The defination of condition number is the max weight over min weight.

        :param start: the start of ref_distance
        :param end: the end of ref_distance
        :param gap: the delta value
        :param plot: plot flag
        :param figname: save the figure to figname
        :return: a list of ref_azimuth and condition numbers
        """
        nscans = int((end - start) / gap) + 1
        ref_azis = \
            [start + gap * i for i in range(nscans)]
        cond_nums = np.zeros(nscans)

        azi_m = self._build_azimuth_matrix()
        for idx, _ref_azi in enumerate(ref_azis):
            weight, _ = self._transfer_azi_to_weight(azi_m, _ref_azi)
            cond_nums[idx] = max(weight) / min(weight)

        if plot:
            plt.plot(ref_azis, cond_nums, 'r-*')
            plt.xlabel("Reference azimuth(degree)")
            plt.ylabel("Condition number")
            if figname is None:
                plt.show()
            else:
                plt.savefig(figname)
        return ref_azis, cond_nums

    def smart_scan(self, max_ratio=0.5, start=0.5, gap=0.5, drop_ratio=0.90,
                   plot=False, figname=None):
        """
        Searching for the ref_azimuth by condition number which satisfy
        our condition. As the ref_distance increase from small values(near
        0), we know that the condition number will first increase, reach
        its maxium and then decrease. The final ref_azimuth will satisfy:
            optimal_cond_number = max_cond_number * max_ratio
        The drop ratio determines the searching end point, which is:
            end_cond_number = max_cond_number * drop_ratio

        :param max_ratio: determine the optimal ref_azimuth(return value)
        :param start: search start point
        :param gap: delta
        :param drop_ratio: determin the search end point
        :param plot: plot flag
        :param figname: figure name
        :return: the optimal ref_azimuth and correspoinding condition number
        """
        azi_m = self._build_azimuth_matrix()

        ref_azis = []
        cond_nums = []

        idx = 0
        _ref_azi = start
        while True:
            weight, _ = self._transfer_azi_to_weight(azi_m, _ref_azi)
            _cond_num = max(weight) / min(weight)
            ref_azis.append(_ref_azi)
            cond_nums.append(_cond_num)
            if idx >= 2 and (_cond_num < drop_ratio * max(cond_nums)):
                break
            if _ref_azi > 200.0:
                raise ValueError("Smart scan error with _ref_dist overflow")
            idx += 1
            _ref_azi += gap

        minv = min(cond_nums)
        maxv = max(cond_nums)
        threshold = minv + max_ratio * (maxv - minv)
        best_idx, best_cond_num = search_for_ratio(cond_nums, threshold)
        best_ref_azi = ref_azis[best_idx]

        logger.info("Best ref_distance and corresponding condition number:"
                    "[%f, %f]" % (best_ref_azi, best_cond_num))

        if plot:
            plt.plot(ref_azis, cond_nums, 'r-*')
            plt.xlabel("Reference distance(degree)")
            plt.ylabel("Condition number")
            plt.plot(best_ref_azi, best_cond_num, 'g*', markersize=10)
            plt.plot([ref_azis[0], ref_azis[-1]], [threshold, threshold],
                     'b--')
            if figname is None:
                plt.show()
            else:
                plt.savefig(figname)

        # calculate weight based on the best ref_dist value
        weight, self.exp_matrix = \
            self._transfer_azi_to_weight(azi_m, best_ref_azi)

        self.points_weights = weight
        self.normalize_weight()

        return best_ref_azi, best_cond_num

    def plot_exp_matrix(self, figname=None):
        plot_2d_matrix(self.exp_matrix,
                       title="Azimuth Exponential Matrix",
                       figname=figname)
