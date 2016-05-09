import spaceweight.util as util
import numpy as np
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
