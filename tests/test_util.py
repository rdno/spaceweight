import spaceweight.util as util
import numpy as np
import numpy.testing as npt
import pytest


def test_search_for_ratio():
    array = np.arange(10)
    assert util.search_for_ratio(array, 4.5) == (4, 4)

    assert util.search_for_ratio(array, 4.1) == (4, 4)

    with pytest.raises(ValueError):
        util.search_for_ratio(array, -1)

    with pytest.raises(ValueError):
        util.search_for_ratio(array, 10)


def test_get_bin_number():
    start = 0
    end = 360
    nbins = 12

    assert util.get_bin_number(0, start, end, nbins) == 0
    assert util.get_bin_number(1.0, start, end, nbins) == 0
    assert util.get_bin_number(60.0, start, end, nbins) == 2
    assert util.get_bin_number(360, start, end, nbins) == 11

    with pytest.raises(ValueError):
        util.get_bin_number(360.1, start, end, nbins)

    with pytest.raises(ValueError):
        util.get_bin_number(-0.1, start, end, nbins)


def test_array_into_bins():
    array = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4])
    start = 0
    end = 4
    nbins = 4
    bins, bins_dict = util.sort_array_into_bins(array, start, end, nbins)
    npt.assert_allclose(bins, [5, 3, 2, 2])
    assert bins_dict == {0: [0, 1, 2, 3, 4], 1: [5, 6, 7],
                         2: [8, 9], 3: [10, 11]}
