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

    _found = False
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
    if data < start or data > end:
        raise ValueError("data(%f) not in range(%f, %f)"
                         % (data, start, end))
    bin_size = (end - start) / nbins
    bin_idx = int(data / bin_size)
    if bin_idx == nbins:
        bin_idx = nbins - 1
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

    for idx, value in bin_dict.items():
        if bins[idx] != len(value):
            raise ValueError("Error in bin sort!")

    return bins, bin_dict


def scale_matrix_by_exp(matrix, ref_value=1.0, order=2.0):
    """
    Rescale the matrix value into exp value:
        new_matrix[i][j] = exp(-(matrix[i][j] / ref_value) ** value)

    :param matrix:
    :param ref_value:
    :param order:
    :return:
    """
    exp_matrix = np.exp(-(matrix / ref_value) ** order)
    sum_on_row = np.sum(exp_matrix, axis=1)
    return exp_matrix, sum_on_row
