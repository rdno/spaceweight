#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class that contains sphere weighting.

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""

from __future__ import print_function, division, absolute_import

import numpy as np
from math import cos, sin
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from obspy.geodetics import locations2degrees
from obspy.geodetics import gps2dist_azimuth

from . import SpherePoint
from .weightbase import WeightBase
from .plot_util import plot_circular_sector, plot_points_in_polar
from .plot_util import plot_rings_in_polar, plot_two_histograms
from .plot_util import plot_2d_matrix
from .util import sort_array_into_bins, scale_matrix_by_exp
from . import logger
from .util import search_for_ratio
from .spherevoronoi import SphericalVoronoi


def _azimuth(lat1, lon1, lat2, lon2):
    """
    The azimuth(unit:degree) starting from point1 to
    point 2 on the sphere
    """
    _, azi, _ = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    return azi


def _distance(lat1, lon1, lat2, lon2):
    """
    The distance(unit:degree) between 2 points on
    the sphere, unit in degree
    """
    return locations2degrees(lat1, lon1, lat2, lon2)


class SphereWeightBase(WeightBase):
    """
    The superclass for sphere weight. It handles 2D sphere problem,
    which means the points should be on the surface of a globe
    """
    def __init__(self, points, center=None, sort_by_tag=False,
                 remove_duplicate=False, normalize_mode="average"):
        """
        :param points: the list of points
        :type points: list
        :param center: the center point. Not required by this weighting.
            If provied, then some methods regarding to azimuth(to the center)
            can be used.
        :param sort_by_tag: refer to superclass WeightBase
        :param remove_duplicate: refer to superclass WeightBase
        :param normalize_mode: refer to superclass WeightBase
        :return:
        """
        if not isinstance(points[0], SpherePoint):
            raise TypeError("Type of points should be SpherePoint")
        if center is not None and not isinstance(center, SpherePoint):
            raise TypeError("Type of center should be SpherePoint")

        WeightBase.__init__(self, points, sort_by_tag=sort_by_tag,
                            remove_duplicate=remove_duplicate)
        self.normalize_mode = normalize_mode
        self.center = center

        if self.points_dimension != (2,):
            raise ValueError("For the sphere problem, dimension of points "
                             "coordinates should be 2: [latitude, logitude].")

    def _calculate_azimuth_array(self):
        if self.center is None:
            raise ValueError("Center must be specified to calculate azimuth")

        azi_array = np.zeros(self.npoints)
        for idx, point in enumerate(self.points):
            azi = _azimuth(self.center.coordinate[0],
                           self.center.coordinate[1],
                           point.coordinate[0],
                           point.coordinate[1])
            azi_array[idx] = azi
        return azi_array

    def _stats_azimuth_info(self, nbins):
        if self.center is None:
            raise ValueError("No center information provided. Impossible to"
                             "calculate azimuth information.")

        azi_array = self._calculate_azimuth_array()

        azi_bin, azi_bin_dict = \
            sort_array_into_bins(azi_array, 0.0, 360.0, nbins=nbins)

        return azi_array, azi_bin, azi_bin_dict

    def _calculate_distance_array(self):
        if self.center is None:
            raise ValueError("No center information provied. Impossible to"
                             "calculate distances from center to points")
        npts = self.npoints
        dist_array = np.zeros(npts)
        for idx, point in enumerate(self.points):
            dist = _distance(self.center.coordinate[0],
                             self.center.coordinate[1],
                             point.coordinate[0],
                             point.coordinate[1])
            dist_array[idx] = dist
        return dist_array

    def _stats_distance_info(self, nbins):
        if self.center is None:
            raise ValueError("No center information provied. Impossible to"
                             "calculate distances from center to points")

        dist_array = self._calculate_distance_array()

        dist_bin, dist_bin_dict = \
            sort_array_into_bins(dist_array, 0.0, 180.0, nbins=nbins)

        return dist_array, dist_bin, dist_bin_dict

    def _sort_weight_into_bins(self, nbins):
        if self.center is None:
            raise ValueError("No event information provided. Impossible to"
                             "calculate azimuth information")

        azi_array, azi_bin, azi_bin_dict = self._stats_azimuth_info(nbins)
        azi_weight_bin = np.zeros(nbins)
        for bin_idx, station_list in azi_bin_dict.iteritems():
            azi_weight_bin[bin_idx] = \
                np.sum(self.points_weights[station_list])

        dist_array, dist_bin, dist_bin_dict = \
            self._stats_distance_info(nbins=nbins)
        dist_weight_bin = np.zeros(nbins)
        for bin_idx, station_list in dist_bin_dict.iteritems():
            dist_weight_bin[bin_idx] = \
                np.sum(self.points_weights[station_list])

        return azi_array, azi_bin, azi_weight_bin, \
            dist_array, dist_bin, dist_weight_bin

    def plot_global_map(self, figname=None, lon0=None):
        """
        Plot global map of points and centers
        """
        from mpl_toolkits.basemap import Basemap

        fig = plt.figure(figsize=(10, 4))

        if lon0 is None:
            if self.center is not None:
                lon0 = self.center.coordinate[1]
            else:
                lon0 = 180.0

        m = Basemap(projection='moll', lon_0=lon0, lat_0=0.0,
                    resolution='c')

        m.drawcoastlines()
        m.fillcontinents()
        m.drawparallels(np.arange(-90., 120., 30.))
        m.drawmeridians(np.arange(0., 420., 60.))
        m.drawmapboundary()

        cm = plt.cm.get_cmap('RdYlBu')
        x, y = m(self.points_coordinates[:, 1], self.points_coordinates[:, 0])
        m.scatter(x, y, 100, color=self.points_weights, marker="^",
                  edgecolor="k", linewidth='0.3', zorder=3, cmap=cm,
                  alpha=0.8)
        plt.colorbar(shrink=0.95)

        if self.center is not None:
            center_lat = self.center.coordinate[0]
            center_lon = self.center.coordinate[1]
            center_x, center_y = m(center_lon, center_lat)
            m.scatter(center_x, center_y, 150, color="g", marker="o",
                      edgecolor="k", linewidth='0.3', zorder=3)

        plt.tight_layout()
        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)
            plt.close(fig)

    def plot_station_weight_distribution(self, nbins=12, figname=None):
        """
        Plot distribution of station and weight in azimuth bins
        """
        if not isinstance(self.center, SpherePoint):
            raise ValueError("No event information provided. Impossible to"
                             "calculate azimuth information")

        azi_array, azi_bin, azi_weight_bin, \
            dist_array, dist_bin, dist_weight_bin = \
            self._sort_weight_into_bins(nbins=nbins)

        fig = plt.figure(figsize=(20, 10))
        g = gridspec.GridSpec(2, 4)

        # plot the stations in polar coords
        plt.subplot(g[0, 0])
        plot_points_in_polar(dist_array, azi_array)

        # plot the station counts in azimuth bins
        plt.subplot(g[0, 1], polar=True)
        plot_circular_sector(azi_bin, title="Points Azimuthal bins")
        # plot the stations weights sum in azimuth bins
        plt.subplot(g[0, 2], polar=True)
        plot_circular_sector(azi_weight_bin,
                             title="Weight sum in Azimuthal bins")

        # plot the histogram of station counts and weights sum in azimuth bins
        plt.subplot(g[0, 3])
        plot_two_histograms(azi_bin, azi_weight_bin, tag1="stations",
                            tag2="weights")

        # plot the stations counts in epi-center distance bins
        plt.subplot(g[1, 1], polar=True)
        bin_edges = np.linspace(0, 180, nbins, endpoint=False)
        plot_rings_in_polar(dist_bin, bin_edges,
                            title="Distance bins")
        # plot the stations weights sum in distance bins
        plt.subplot(g[1, 2], polar=True)
        bin_edges = np.linspace(0, 180, nbins, endpoint=False)
        plot_rings_in_polar(dist_weight_bin, bin_edges,
                            title="Weight in distance bin")
        # plot the histogram of station counts and weights in distance bins
        plt.subplot(g[1, 3])
        bin_edges = np.linspace(0, 180, nbins, endpoint=False)
        plot_two_histograms(dist_bin, dist_weight_bin, bin_edges,
                            tag1="stations", tag2="weights")

        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)
            plt.close(fig)


class SphereDistRel(SphereWeightBase):
    """
    Class that using the distances between points to calculate the weight.
    The basic idea is if two points are close to each other, the contribution
    to weight will be high(weight will be small).
    """

    def __init__(self, points, center=None, sort_by_tag=False,
                 remove_duplicate=False, normalize_mode="average"):

        SphereWeightBase.__init__(self, points, center=center,
                                  sort_by_tag=sort_by_tag,
                                  remove_duplicate=remove_duplicate,
                                  normalize_mode=normalize_mode)

        self.exp_matrix = np.zeros([self.npoints, self.npoints])

    def _build_distance_matrix(self):
        """
        calculate distance matrix
        """
        coords = self.points_coordinates
        npts = self.npoints
        dist_m = np.zeros([npts, npts])
        # calculate the upper part
        for _i in range(npts):
            for _j in range(_i+1, npts):
                loc_i = coords[_i]
                loc_j = coords[_j]
                dist_m[_i, _j] = \
                    _distance(loc_i[0], loc_i[1],
                              loc_j[0], loc_j[1])
                # symetric
                dist_m[_j, _i] = dist_m[_i, _j]
        # fill dianogal with zero, which is the station self term
        np.fill_diagonal(dist_m, 0.0)
        return dist_m

    @staticmethod
    def _transfer_dist_to_weight(dist_m, ref_distance):
        """
        Transfer the distance matrix into weight matrix by a given
        reference distance(distance unit is degree)

        :param dist_m:
        :param ref_distance:
        :return:
        """
        exp_matrix, sum_on_row = scale_matrix_by_exp(dist_m, ref_distance,
                                                     order=2.0)
        weight = 1. / sum_on_row
        return weight, exp_matrix

    def calculate_weight(self, ref_distance):
        """
        Calculate the weight based upon a given reference distance

        :param ref_distance:
        :return:
        """
        """
        :param ref_distance:
        :return:
        """
        dist_m = self._build_distance_matrix()

        weight, self.exp_matrix = \
            self._transfer_dist_to_weight(dist_m, ref_distance)

        self.points_weights = weight
        self.normalize_weight()

        logger.info("Number of points at this stage: %d" % self.npoints)
        logger.info("Condition number of weight array(max/min): %8.2f"
                    % self.condition_number)

    def scan(self, start=1.0, end=50.0, gap=1.0, plot=False, figname=None):
        """
        Scan among the range of ref_dists and return the condition number
        The defination of condition number is the max weight over min weight.

        :param start: the start of ref_distance
        :param end: the end of ref_distance
        :param gap: the delta value
        :param plot: plot flag
        :param figname: save the figure to figname
        :return: a list of ref_distance and condition numbers
        """

        nscans = int((end - start) / gap) + 1
        ref_dists = \
            [start + gap * i for i in range(nscans)]
        cond_nums = np.zeros(nscans)

        dist_m = self._build_distance_matrix()
        for idx, _ref_dist in enumerate(ref_dists):
            weight, _ = self._transfer_dist_to_weight(dist_m, _ref_dist)
            cond_nums[idx] = max(weight) / min(weight)

        if plot:
            plt.plot(ref_dists, cond_nums, 'r-*')
            plt.xlabel("Reference distance(degree)")
            plt.ylabel("Condition number")
            if figname is None:
                plt.show()
            else:
                plt.savefig(figname)
        return ref_dists, cond_nums

    def smart_scan(self, max_ratio=0.5, start=1.0, gap=0.5, drop_ratio=0.20,
                   plot=False, figname=None):
        """
        Searching for the ref_distance by condition number which satisfy
        our condition. As the ref_distance increase from small values(near
        0), we know that the condition number will first increase, reach
        its maxium and then decrease. The final ref_distance will satisfy:
            optimal_cond_number = max_cond_number * max_ratio
        The drop ratio determines the searching end point, which is:
            end_cond_number = max_cond_number * drop_ratio

        :param max_ratio: determine the optimal ref_distance(return value)
        :param start: search start point
        :param gap: delta
        :param drop_ratio: determin the search end point
        :param plot: plot flag
        :param figname: figure name
        :return: the optimal ref_distance and correspoinding condition number
        """
        # print("npoints: %d" % self.npoints)
        if self.npoints <= 2:
            # if only two points, then the all the weights will be 1
            # anyway
            logger.info("Less than two points so the weights are automatically"
                        "set to 1")
            self.points_weights = np.ones(self.npoints)
            self.normalize_weight()
            return None, 1

        if self.npoints <= 10:
            # reset drop ratio if there is less that 5 points
            # otherwise, it might go overflow while searching for
            # drop. Note that this will not impact the final search
            # result.
            drop_ratio = 0.99

        dist_m = self._build_distance_matrix()

        ref_dists = []
        cond_nums = []

        idx = 0
        _ref_dist = start
        while True:
            weight, _ = self._transfer_dist_to_weight(dist_m, _ref_dist)
            _cond_num = max(weight) / min(weight)
            ref_dists.append(_ref_dist)
            cond_nums.append(_cond_num)
            if idx >= 2 and (_cond_num < drop_ratio * max(cond_nums)):
                break
            if _ref_dist > 200.0:
                if np.isclose(max(cond_nums), min(cond_nums)):
                    print("cond nums are very close to each other")
                    break
                else:
                    print("Smart scan error with _ref_dist overflow")
                    return None, None
            idx += 1
            _ref_dist += gap

        minv = min(cond_nums)
        maxv = max(cond_nums)
        threshold = minv + max_ratio * (maxv - minv)
        best_idx, best_cond_num = search_for_ratio(cond_nums, threshold)
        best_ref_dist = ref_dists[best_idx]

        logger.info("Best ref_distance and corresponding condition number:"
                    "[%f, %f]" % (best_ref_dist, best_cond_num))

        if plot:
            plt.plot(ref_dists, cond_nums, 'r-*')
            plt.xlabel("Reference distance(degree)")
            plt.ylabel("Condition number")
            plt.plot(best_ref_dist, best_cond_num, 'g*', markersize=10)
            plt.plot([ref_dists[0], ref_dists[-1]], [threshold, threshold],
                     'b--')
            if figname is None:
                plt.show()
            else:
                plt.savefig(figname)

        # calculate weight based on the best ref_dist value
        weight, self.exp_matrix = \
            self._transfer_dist_to_weight(dist_m, best_ref_dist)

        self.points_weights = weight
        self.normalize_weight()

        return best_ref_dist, best_cond_num

    def plot_exp_matrix(self, figname=None):
        plot_2d_matrix(self.exp_matrix,
                       title="Distance Exponential Matrix",
                       figname=figname)


class SphereVoronoi(SphereWeightBase):

    def __init__(self, points, voronoi_order=1.0, center=None,
                 sort_by_tag=False, remove_duplicate=True,
                 normalize_mode="average"):
        """
        :param points: a list of SpherePoints
        :param voronoi_order: voronoi order. The weight is determined by
            by the surface area of voronoi cell, to a cetain order:
                weight = (surface_area) ** voronoi_order
        :param center: center point
        :param sort_by_tag:
        :param remove_duplicate:
        :param normalize_mode:
        :return:
        """
        SphereWeightBase.__init__(self, points, center=center,
                                  sort_by_tag=sort_by_tag,
                                  remove_duplicate=remove_duplicate,
                                  normalize_mode=normalize_mode)
        self.sv = None
        self.voronoi_order = voronoi_order

        # sphere parameter for voronoi usage
        self.sphere_radius = 1.0
        self.sphere_center = np.zeros(3)

    def _transfer_coordinate(self):
        """
        Transfer (longitude, latitude) to (x, y, z) on sphere(radius, center)
        """
        radius = self.sphere_radius
        center = self.sphere_center
        sphere_loc = np.zeros([self.npoints, 3])

        for _i, point in enumerate(self.points):
            lat = np.deg2rad(point.coordinate[0])
            lon = np.deg2rad(point.coordinate[1])
            sphere_loc[_i, 0] = radius * cos(lat) * cos(lon) + center[0]
            sphere_loc[_i, 1] = radius * cos(lat) * sin(lon) + center[1]
            sphere_loc[_i, 2] = radius * sin(lat) + center[2]

        return sphere_loc

    def calculate_weight(self):
        trans_points = self._transfer_coordinate()
        # for _i in range(self.nstations):
        #    print("%10s: [%10.5f, %10.5f] -- [%10.5f, %10.5f, %10.5f]"
        #          % (self.station_tag[_i], self.station_loc[_i][0],
        #             self.station_loc[_i][1], self.points[_i][0],
        #             self.points[_i][1], self.points[_i][2]))

        self.sv = SphericalVoronoi(trans_points, radius=self.sphere_radius,
                                   center=self.sphere_center)
        self.sv.sort_vertices_of_regions()

        surface_area, coverage = self.sv.compute_surface_area()
        weight = surface_area ** self.voronoi_order

        logger.info("Voronoi surface area coverage: %15.5f" % coverage)
        self.points_weights = weight
        self.normalize_weight()

        logger.info("Number of points at this stage: %d" % self.npoints)
        logger.info("Condition number of weight array(max/min): %8.2f"
                    % self.condition_number)

    def plot_sphere(self):
        points = self._transfer_coordinate()
        sv = self.sv
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # plot the unit sphere for reference (optional)
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='y', alpha=0.05)
        # plot Voronoi vertices
        # ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2],
        #           c='g')
        # indicate Voronoi regions (as Euclidean polygons)
        for region in sv.regions:
            random_color = colors.rgb2hex(np.random.rand(3))
            polygon = Poly3DCollection([sv.vertices[region]], alpha=1.0)
            polygon.set_color(random_color)
            ax.add_collection3d(polygon)

        # plot generator points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xticks([-1, 1])
        ax.set_yticks([-1, 1])
        ax.set_zticks([-1, 1])
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.xlabel("X")
        plt.ylabel("Y")

        plt.show()
