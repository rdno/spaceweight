from __future__ import print_function, division

import numpy as np
from spaceweight import SpherePoint
from spaceweight import SphereAziRel, SphereAziBin


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
        point = SpherePoint(lat, lon, tag)
        points.append(point)

    return points


def test_sphereazibin(points, center, bin_order=0.5, nbins=12):

    weight = SphereAziBin(points, center=center, bin_order=bin_order,
                          nbins=nbins)
    weight.calculate_weight()
    weight.plot_global_map()
    weight.plot_weight_histogram()


def test_sphereazirel(points, center, ref_azi=5.0):

    weight = SphereAziRel(points, center=center)
    ref_azi, _ = weight.smart_scan(plot=True)
    weight.calculate_weight(ref_azi)
    weight.plot_exp_matrix()
    weight.plot_global_map()
    weight.plot_weight_histogram()
    #weight.scan(plot=True)
    # weight.write_weight(filename="azirel.txt")
    # weight.plot_station_weight_distribution(nbins=12)
    weight.calculate_weight(30)
    weight.plot_exp_matrix()
    weight.plot_global_map()


if __name__ == "__main__":
    points = read_points("./STATIONS_waveforms")
    center = SpherePoint(0, 0, "center")

    test_sphereazibin(points, center)
    test_sphereazirel(points, center)

