"""
Utils
"""
import numpy as np


def search_for_ratio(array, threshold):
    """
    Point by point search yarray[i] ~= threshold

    :param array:
    :param threshold:
    :return: the index of array element, and the value
    """
    best_idx = None
    if threshold > max(array) or threshold < min(array):
        raise ValueError("threshold(%f) is out of array range(%f, %f)"
                         % (threshold, min(array), max(array)))

    for i in range(len(array) - 1):
        if (array[i] - threshold) * (array[i + 1] - threshold) <= 0:
            _found = True
            break

    if not _found:
        return None, None

    if abs(array[i] - threshold) <= abs(array[i + 1] - threshold):
        best_idx = i
    else:
        best_idx = i + 1
    return best_idx, array[best_idx]


def get_bin_number(data, start, end, nbins):
    """
    Calculate the bin number of given start, end and number of bins
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


def sort_array_into_bins(array, start, end, nbins):
    if start > min(array):
        raise ValueError("Sort array into bins error: %d < %d" %
                         (start, min(array)))
    if end < max(array):
        raise ValueError("Sort array into bins error: %d < %d" %
                         (max(array), end))
    bin_edge = np.linspace(start, end, nbins + 1)
    bins = np.histogram(array, bin_edge)[0]

    bin_dict = dict()
    for _i in range(nbins):
        bin_dict[_i] = []

    for data_idx, data in enumerate(array):
        bin_idx = get_bin_number(data, start, end, nbins)
        bin_dict[bin_idx].append(data_idx)

    for idx, value in bin_dict.iteritems():
        if bins[idx] != len(value):
            raise ValueError("Error in bin sort!")

    return bins, bin_dict
