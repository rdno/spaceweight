"""
Utils
"""


def search_for_ratio(xarray, yarray, threshold):
    if len(xarray) != len(yarray):
        raise ValueError("Length of xarray(%d) and yarray(%d) not the same!"
                         % (len(xarray), len(yarray)))

    if threshold > max(yarray) or threshold < min(yarray):
        raise ValueError("threshold(%f) is out of yarray range(%f, %f)"
                         % (threshold, min(yarray), max(yarray)))

    for i in range(len(xarray) - 1):
        if (yarray[i] - threshold) * (yarray[i+1] - threshold) <= 0:
            _found = True
            break

    if _found:
        return xarray[i], yarray[i]
    else:
        return None, None
