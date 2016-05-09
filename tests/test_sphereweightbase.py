from __future__ import print_function, division

import os
import inspect
import pytest
import matplotlib
matplotlib.use("Agg")  # NOQA

import numpy as np
import numpy.testing as npt

from spaceweight import Point
from spaceweight.weightbase import WeightBase

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(
     inspect.currentframe())))


def parse_station_file(filename):
    pts = []
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
        pt = Point(coord, tag)
        pts.append(pt)

    return pts


@pytest.fixture
def lpoints():
    POINTSFILE = os.path.join(CURRENTDIR, "data", "STATIONS")
    return parse_station_file(POINTSFILE)


@pytest.fixture
def spoints():
    POINTSFILE = os.path.join(CURRENTDIR, "data", "STATIONS_SMALL")
    return parse_station_file(POINTSFILE)


def test_weightbase(spoints):
    wobj = WeightBase(spoints, sort_by_tag=True, remove_duplicate=True)
    assert wobj.points_dimension == (2,)
    assert wobj.npoints == 5
    assert wobj.points_tags == ["AA_BRVK", "II_AAK", "II_ASCN", "II_BFO",
                                "IU_BORG"]
    assert wobj.condition_number is None
    npt.assert_allclose(wobj.points_weights, np.zeros(wobj.npoints))


def test_detect_duplicate_tags(spoints):
    spoints.append(spoints[0])
    with pytest.raises(ValueError) as excinfo:
        WeightBase(spoints, sort_by_tag=True, remove_duplicate=True)
        assert 'Duplicate tags' in excinfo


def test_detect_duplicate_coordinates(spoints):
    extra_point = spoints[0]
    extra_point.tag = "ZZ_EXTRA"
    spoints.append(extra_point)
    with pytest.raises(ValueError):
        wobj = WeightBase(spoints, sort_by_tag=True, remove_duplicate=True)
        assert wobj.npoints == 5
        assert len(wobj.removed_points) == 1


def test_points_weights_setter(spoints):
    wobj = WeightBase(spoints, sort_by_tag=True, remove_duplicate=True)
    weights = np.arange(1, 6)
    wobj.points_weights = weights
    npt.assert_allclose(wobj.points_weights, weights)


def test_normalize_weight(spoints):

    for point in spoints:
        point.weight = 5
    wobj = WeightBase(spoints, sort_by_tag=True, remove_duplicate=True)

    wobj.normalize_weight(mode="average")
    npt.assert_allclose(wobj.points_weights, np.ones(5))

    wobj.normalize_weight(mode="sum")
    npt.assert_allclose(wobj.points_weights, 0.2 * np.ones(5))

    wobj.normalize_weight(mode="max")
    npt.assert_allclose(wobj.points_weights, np.ones(5))


def test_plot_weight_histogram(spoints, tmpdir):
    for point in spoints:
        point.weight = 5
    wobj = WeightBase(spoints, sort_by_tag=True, remove_duplicate=True)

    figname = os.path.join(str(tmpdir), "hist.png")
    wobj.plot_weight_histogram(figname)
