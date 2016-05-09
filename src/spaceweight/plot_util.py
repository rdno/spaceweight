import matplotlib.pyplot as plt
import numpy as np


def plot_2d_matrix(matrix, title="", figname=None):
    """
    plot 2D matrix with color bar
    :param matrix:
    :param title:
    :param figname:
    :return:
    """
    fig = plt.figure(figsize=(12, 6.4))

    ax = fig.add_subplot(111)
    ax.set_title('colorMap: %s' % title)
    plt.imshow(matrix, interpolation='none')
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)
        plt.close(fig)


def plot_points_in_polar(rads, azis, mode="global",
                         title="Points distribution"):
    """
    Plot the points in polar axis if given radius and azimuths

    :param rads:
    :param azis:
    :param mode:
    :param title:
    :return:
    """
    plt.axes(projection="polar")
    c = plt.scatter(azis, rads, marker=u'^', c='r',
                    s=20, edgecolor='k', linewidth='0.3')
    c.set_alpha(0.75)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=6)
    plt.title(title, fontsize=10)

    ax = plt.gca()
    if mode == "regional":
        ax.set_rmax(1.10 * max(rads))
    elif mode == "global":
        ax.set_rmax(180)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)


def plot_circular_sector(csectors, title="Circular Sector"):
    """
    plot circular sectors in polar axis

    :param csectors:
    :param title:
    :return:
    """
    # set plt.subplot(***, polar=True)
    plt.axes(projection="polar")
    if title is not None:
        plt.title(title, fontsize=10)

    nbins = len(csectors)
    delta = 2 * np.pi / nbins
    bins = [delta * i for i in range(nbins)]
    norm_factor = np.max(csectors)

    bars = plt.bar(bins, csectors, width=delta, bottom=0.0)
    for r, bar in zip(csectors, bars):
        bar.set_facecolor(plt.cm.jet(r / norm_factor))
        bar.set_alpha(0.8)
        bar.set_linewidth(0.3)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=6)

    ax = plt.gca()
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)


def plot_rings_in_polar(ring_bins, ring_edges, title=""):
    """
    Plot rings in polar axis given ring bins and ring edges

    :param ring_bins:
    :param ring_edges:
    :param title:
    :return:
    """
    if len(ring_bins) != (len(ring_edges) - 1):
        raise ValueError("Length of ring bins and edges must be the same")

    plt.axes(projection="polar")
    theta = np.linspace(0., 2.*np.pi, 80, endpoint=True)
    norm_factor = max(ring_bins)

    ax = plt.gca()
    for _i in range(len(ring_bins)):
        color = plt.cm.jet(ring_bins[_i]/norm_factor)
        ax.fill_between(theta, ring_edges[_i], ring_edges[_i+1],
                        color=color, alpha=0.8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=6)
    plt.title(title, fontsize=10)


def plot_two_histograms(array1, array2, bin_edges=None,
                        tag1="array1", tag2="array2"):
    nbins = len(array1)

    if nbins != len(array2):
        raise ValueError("Length of array1(%d) not equal to array2(%d)"
                         % (len(array1), len(array2)))
    if bin_edges is None:
        bin_edges = range(nbins)
    else:
        if nbins != len(bin_edges):
            raise ValueError("length of array(%d) not equal as bin left egdes"
                             "%d" % (len(array1), len(bin_edges)))

    array1 = array1 / sum(array1)
    array2 = array2 / sum(array2)
    width = bin_edges[1] - bin_edges[0]

    plt.bar(bin_edges, array1, alpha=0.5, width=width, label=tag1,
            color="g")
    plt.bar(bin_edges, array2, alpha=0.5, width=width, label=tag2,
            color="b")
    plt.legend()
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=6)
    plt.xlim([bin_edges[0], bin_edges[-1] + width])
