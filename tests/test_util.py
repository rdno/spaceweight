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


def get_bin_number():
    pass
