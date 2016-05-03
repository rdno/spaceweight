#!/usr/bin/env python

"""
Contains several distance and azimuth weighting strategy on
 surface of a sphere
 Distance: 1) Relative Distance Weighting
           2) Voronoi Weighting(a slighting modified version of:
              https://github.com/tylerjereddy/py_sphere_Voronoi)
           3) Exponential Distance Weighting(from CMT3D, by Qinya Liu)
 Azimuth:  1) Relative Azimuth Weighting
           2) Bin Azimuth Weighting(from CMT3D, by Qinya Liu)
"""

from __future__ import print_function, division, absolute_import

import numpy as np
from math import cos, sin
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap

from obspy.geodetics import locations2degrees
from obspy.geodetics import gps2dist_azimuth

from . import SpherePoint
from . import logger
from .spherevoronoi import SphericalVoronoi
from .weightbase import WeightBase


class SphereWeightBase(WeightBase):
    """
    2D sphere problem, which means the points should be
    on the surface of a globe
    """
    def __init__(self, points, center=None, remove_duplicate=False,
                 normalize_flag=True, normalize_mode="average"):
        WeightBase.__init__(self, points, remove_duplicate=remove_duplicate,
                            normalize_flag=normalize_flag,
                            normalize_mode=normalize_mode)

        self.center = center
        if self.dimension != 2:
            raise ValueError("For the sphere problem, dimension of points "
                             "should be 2: [latitude, logitude].")

    @staticmethod
    def _azimuth(lat1, lon1, lat2, lon2):
        """
        The azimuth(unit:degree) starting from point1 to
        point 2
        """
        _, azi, _ = gps2dist_azimuth(lat1, lon1, lat2, lon2)
        return azi

    @staticmethod
    def _distance(lat1, lon1, lat2, lon2):
        """
        The distance(unit:degree) between 2 points on
        the sphere
        """
        return locations2degrees(lat1, lon1, lat2, lon2)

    def _calculate_azimuth_array(self):
        if self.center is None:
            raise ValueError("Center must be specified to calculate azimuth")

        npts = self.npoints
        azi_array = np.zeros(npts)
        for idx, point in enumerate(self.points):
            azi = self._azimuth(self.center.coordinate[0],
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
            self.sort_array_into_bins(azi_array, 0.0, 360.0, nbins=nbins)

        return azi_array, azi_bin, azi_bin_dict

    def _calculate_distance_array(self):
        if self.center is None:
            raise ValueError("No center information provied. Impossible to"
                             "calculate distances from center to points")
        npts = self.npoints
        dist_array = np.zeros(npts)
        for idx, point in enumerate(self.points):
            dist = self._distance(self.center.coordinate[0],
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
            self.sort_array_into_bins(dist_array, 0.0, 180.0, nbins=nbins)

        return dist_array, dist_bin, dist_bin_dict

    def sort_array_into_bins(self, array, start, end, nbins):
        if start > min(array):
            raise ValueError("Sort array into bins error: %d < %d" %
                             (start, min(array)))
        if end < max(array):
            raise ValueError("Sort array into bins error: %d < %d" %
                             (max(array), end))
        bin_edge = np.linspace(start, end, nbins+1)
        bins = np.histogram(array, bin_edge)[0]

        bin_dict = dict()
        for _i in range(nbins):
            bin_dict[_i] = []

        for data_idx, data in enumerate(array):
            bin_idx = self.get_bin_number(data, start, end, nbins)
            bin_dict[bin_idx].append(data_idx)

        for idx, value in bin_dict.iteritems():
            if bins[idx] != len(value):
                raise ValueError("Error in bin sort!")

        return bins, bin_dict

    @staticmethod
    def get_bin_number(data, start, end, nbins):
        """
        Calculate the bin number of a given azimuth
        :param azimuth: test test test
        :return:
        """
        bin_size = (end - start) / nbins
        bin_idx = int(data / bin_size)
        if bin_idx == nbins:
            bin_idx = nbins - 1
            return bin_idx
        elif bin_idx > nbins:
            raise ValueError("bin idx larger than nbins: %d < %d" %
                             (bin_idx, nbins))
        return bin_idx

    @staticmethod
    def _plot_matrix(matrix, title="", figname=None):

        fig = plt.figure(figsize=(12, 6.4))

        ax = fig.add_subplot(111)
        ax.set_title('colorMap: %s' % title)
        plt.imshow(matrix, interpolation='none')
        ax.set_aspect('equal')

        cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.patch.set_alpha(0)
        cax.set_frame_on(False)
        plt.colorbar(orientation='vertical')

        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)
            plt.close(fig)

    def plot_global_map(self, figname=None):
        """
        Plot global map of points and centers
        """
        fig = plt.figure(figsize=(10, 4))

        if self.center is None:
            m = Basemap(projection='moll', lon_0=180.0, lat_0=0.0,
                        resolution='c')
        else:
            lon_0 = self.center.coordinate[1]
            m = Basemap(projection='moll', lon_0=lon_0, lat_0=0.0,
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

    def plot_azimuth_distance_distribution(self, nbins=12, figname=None):
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
        plt.subplot(g[0, 0], polar=True)
        self._plot_points_distribution(dist_array, azi_array)
        plt.subplot(g[0, 1], polar=True)
        self._plot_circular_sector(
            azi_bin, title="Points Azimuthal bins")
        plt.subplot(g[0, 2], polar=True)
        self._plot_circular_sector(azi_weight_bin,
                                   title="Weight in Azimuthal bins")
        plt.subplot(g[0, 3])
        self._plot_two_histograms(azi_bin, azi_weight_bin, 0, 360, nbins)

        plt.subplot(g[1, 1], polar=True)
        self._plot_rings_in_polar(dist_bin, 0, 180, nbins,
                                  title="Distance bins")
        plt.subplot(g[1, 2], polar=True)
        self._plot_rings_in_polar(dist_weight_bin, 0, 180, nbins,
                                  title="Weight in distance bin")
        plt.subplot(g[1, 3])
        self._plot_two_histograms(dist_bin, dist_weight_bin, 0, 180, nbins)

        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)
            plt.close(fig)

    @staticmethod
    def _plot_points_distribution(sta_dist, sta_theta, mode="global",
                                  title="Points distribution"):
        ax = plt.gca()
        c = plt.scatter(sta_theta, sta_dist, marker=u'^', c='r',
                        s=20, edgecolor='k', linewidth='0.3')
        c.set_alpha(0.75)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=6)
        if mode == "regional":
            ax.set_rmax(1.10 * max(sta_dist))
        elif mode == "global":
            ax.set_rmax(180)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        plt.title(title, fontsize=10)

    @staticmethod
    def _plot_circular_sector(circular_bin, title=None):
        # set plt.subplot(***, polar=True)
        if title is not None:
            plt.title(title, fontsize=10)

        nbins = len(circular_bin)
        delta = 2*np.pi/nbins
        bins = [delta*i for i in range(nbins)]
        norm_factor = np.max(circular_bin)

        bars = plt.bar(bins, circular_bin, width=delta, bottom=0.0)
        for r, bar in zip(circular_bin, bars):
            bar.set_facecolor(plt.cm.jet(r/norm_factor))
            bar.set_alpha(0.8)
            bar.set_linewidth(0.3)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=6)
        ax = plt.gca()
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    @staticmethod
    def _plot_rings_in_polar(ring_bin, start, end, nbins,
                             title="Distance bin"):
        ax = plt.gca()
        theta = np.linspace(0., 2.*np.pi, 80, endpoint=True)
        ring_edge = np.linspace(start, end, nbins+1)
        norm_factor = max(ring_bin)
        for _i in range(nbins):
            color = plt.cm.jet(ring_bin[_i]/norm_factor)
            ax.fill_between(theta, ring_edge[_i], ring_edge[_i+1],
                            color=color, alpha=0.8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=6)
        plt.title(title, fontsize=10)

    @staticmethod
    def _plot_two_histograms(bin_array, weight_array, start, end, nbins):
        barray = bin_array / sum(bin_array)
        warray = weight_array / sum(weight_array)

        width = (end - start) / nbins
        bin_edges = [i*width for i in range(nbins)]

        plt.bar(bin_edges, barray, alpha=0.5, width=width, label="bin",
                color="g")
        plt.bar(bin_edges, warray, alpha=0.5, width=width,
                label="weight", color="b")
        plt.legend()
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=6)
        plt.xlim([start, end])


class SphereDistRel(SphereWeightBase):

    def __init__(self, points, ref_distance=1.0, center=None,
                 remove_duplicate=False,
                 normalize_flag=True, normalize_mode="average"):

        SphereWeightBase.__init__(self, points, center=center,
                                  remove_duplicate=remove_duplicate,
                                  normalize_flag=normalize_flag,
                                  normalize_mode=normalize_mode)

        self.ref_distance = ref_distance

        self.exp_matrix = np.zeros([self.npoints, self.npoints])

    def _build_distance_matrix(self):
        """
        calculate distance matrix
        """
        coords = self.points_coordinates
        npts = self.npoints
        dist_m = np.zeros([self.npoints, self.npoints])
        # calculate the upper part
        for _i in range(npts):
            for _j in range(_i+1, npts):
                loc_i = coords[_i]
                loc_j = coords[_j]
                dist_m[_i, _j] = \
                    self._distance(loc_i[0], loc_i[1],
                                   loc_j[0], loc_j[1])
                # symetric
                dist_m[_j, _i] = dist_m[_i, _j]
        # fill dianogal again
        # np.fill_diagonal(dist_m, 0.0)
        return dist_m

    def calculate_weight(self):
        dist_m = self._build_distance_matrix()

        self.exp_matrix = np.exp(-(dist_m / self.ref_distance)**2)

        sum_exp = np.sum(self.exp_matrix, axis=1)
        weight = 1. / sum_exp

        self._set_points_weights(weight)
        self._normalize_weight()

        logger.info("Number of points at this stage: %d" % self.npoints)
        logger.info("Condition number of weight array(max/min): %8.2f"
                    % self.condition_number)

    def plot_exp_matrix(self, figname=None):
        self._plot_matrix(self.exp_matrix,
                          title="Distance Exponential Matrix",
                          figname=figname)


class SphereVoronoi(SphereWeightBase):

    def __init__(self, points, voronoi_order=1.0, center=None,
                 remove_duplicate=True,
                 normalize_flag=True, normalize_mode="average"):

        SphereWeightBase.__init__(self, points, center=center,
                                  remove_duplicate=remove_duplicate,
                                  normalize_flag=normalize_flag,
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
        self._set_points_weights(weight)
        self._normalize_weight()
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


class SphereAziBin(SphereWeightBase):
    """
    Used in original version of CMT3D(Qinya Liu).
    Epicenter distance exponetial weighting strategy.
    """

    def __init__(self, points, bin_order=0.5, nbins=12, center=None,
                 remove_duplicate=False, normalize_flag=True,
                 normalize_mode="average"):
        """
        :param ref_distance: reference epicenter distance(unit: degree)
        """
        if not isinstance(center, SpherePoint):
            raise TypeError("For SphereAziBin center must be specified;"
                            "Otherwise, Azimuth can not be measured")
        SphereWeightBase.__init__(self, points, center=center,
                                  remove_duplicate=remove_duplicate,
                                  normalize_flag=normalize_flag,
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

        self._set_points_weights(weight)
        self._normalize_weight()

        logger.info("Number of points at this stage: %d" % self.npoints)
        logger.info("Condition number of weight array(max/min): %8.2f"
                    % self.condition_number)


class SphereAziRel(SphereWeightBase):

    def __init__(self, points, ref_azimuth=1.0, center=None,
                 remove_duplicate=False, normalize_flag=True,
                 normalize_mode="average"):
        """
        :param ref_distance: reference epicenter distance(unit: degree)
        """
        if not isinstance(center, SpherePoint):
            raise TypeError("For SphereAziBin center must be specified;"
                            "Otherwise, Azimuth can not be measured")
        SphereWeightBase.__init__(self, points, center=center,
                                  remove_duplicate=remove_duplicate,
                                  normalize_flag=normalize_flag,
                                  normalize_mode=normalize_mode)
        self.ref_azimuth = ref_azimuth

    def calculate_weight(self):

        weight = np.zeros(self.npoints)

        azi_matrix = self._build_azimuth_matrix()

        self.exp_matrix = np.exp(-(azi_matrix / self.ref_azimuth)**2)

        sum_azi_matrix = np.sum(self.exp_matrix, axis=1)
        weight = 1.0 / sum_azi_matrix

        self._set_points_weights(weight)
        self._normalize_weight()

        logger.info("Number of points at this stage: %d" % self.npoints)
        logger.info("Condition number of weight array(max/min): %8.2f"
                    % self.condition_number)

    def _build_azimuth_matrix(self):

        azi_array = self._calculate_azimuth_array()

        npts = self.npoints
        azi_matrix = np.zeros([npts, npts])
        for _i in range(npts):
            for _j in range(_i+1, npts):
                azi_matrix[_i, _j] = azi_array[_i] - azi_array[_j]
                azi_matrix[_j, _i] = - azi_matrix[_i, _j]

        # np.fill_diagonal(azi_m, 0.0)
        return azi_matrix

    def plot_exp_matrix(self, figname=None):
        self._plot_matrix(self.exp_matrix,
                          title="Azimuth Exponential Matrix",
                          figname=figname)
