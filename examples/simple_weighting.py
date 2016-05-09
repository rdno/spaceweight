from __future__ import print_function, division

import numpy as np
from spaceweight import Point
from spaceweight.weightbase import WeightBase
from spaceweight.sphereweight import SphereWeightBase
from spaceweight import SphereDistRel, SphereAziBin
from spaceweight import SphereAziRel, SphereVoronoi


def read_points(filename):
    points = []
    with open(filename) as fh:
        content = fh.readlines()

    for line in content:
        info = line.split()
        sta = info[0]
        nw = info[1]
        tag = "%s_%s" % (nw, sta)
        lat = float(info[2])
        lon = float(info[3])
        coord = np.array([lat, lon])
        point = Point(coord, tag)
        points.append(point)

    return points


def test_weightbase(points):
    WeightBase(points, sort_by_tag=True, remove_duplicate=True)


def test_sphereweightbase(points, center):

    weight = SphereWeightBase(points, center)
    weight.plot_global_map()


def test_spheredistrel(points, center):

    outputdir = "test23/"
    weight = SphereDistRel(points)
    #weight.calculate_weight(23.0)
    #weight.scan(start=0.1, end=50.0, gap=1.0, plot=True,
    #            figname="scan.png")
    weight.smart_scan(plot=True)
    weight.plot_exp_matrix(outputdir + "exp_matrx.png")
    weight.plot_global_map(outputdir + "map.png")
    weight.plot_weight_histogram(outputdir + "weight_hist.png")


def test_spherevoronoi(points, center, order=1.0):

    weight = SphereVoronoi(points, voronoi_order=order, center=center)
    weight.calculate_weight()
    weight.plot_global_map()
    weight.plot_weight_histogram()


def test_sphereazibin(points, center, bin_order=0.5, nbins=12):

    weight = SphereAziBin(points, center=center, bin_order=bin_order,
                          nbins=nbins)
    weight.calculate_weight()
    weight.plot_global_map()
    weight.plot_weight_histogram()


def test_sphereazirel(points, center, ref_azi=5.0):

    weight = SphereAziRel(points, center=center, ref_azimuth=ref_azi)
    weight.calculate_weight()
    # weight.plot_exp_matrix()
    # weight.plot_global_map()
    # weight.plot_weight_histogram()
    # weight.write_weight(filename="azirel.txt")
    weight.plot_azimuth_distance_distribution(nbins=12)


if __name__ == "__main__":
    points = read_points("./STATIONS_waveforms")
    center = Point([0, 0], "center")

    # test_weightbase(points)
    # test_sphereweightbase(points, center)
    test_spheredistrel(points, center)
    # test_spherevoronoi(points, center, order=1.0)
    # test_sphereazibin(points, center)
    # test_sphereazirel(points, center)
