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


def test_scale_matrix_by_exp():
    matrix = np.array([[1.0, 2.0], [3.0, 0.0]])

    exp_matrix, sum_on_row = util.scale_matrix_by_exp(matrix, ref_value=2.0,
                                                      order=2.0)

    true_m = np.array([[np.exp(-0.25), np.exp(-1.0)],
                       [np.exp(-2.25), np.exp(0)]])
    npt.assert_allclose(exp_matrix, true_m)
    npt.assert_allclose(sum_on_row, [np.sum(true_m[0, :]),
                                     np.sum(true_m[1, :])])
