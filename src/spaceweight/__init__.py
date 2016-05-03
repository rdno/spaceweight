#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import numpy as np

__version__ = "0.1.0"

# Setup the logger.
logger = logging.getLogger("spaceweighting")
logger.setLevel(logging.INFO)
# Prevent propagating to higher loggers.
logger.propagate = 0
# Console log handler.
ch = logging.StreamHandler()
# Add formatter
FORMAT = "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)


class Point(object):
    """ General definication of a point """

    def __init__(self, coordinate, tag, weight=0.0):
        self.coordinate = np.array(coordinate)
        self.tag = tag
        self.weight = weight

        self._sanity_check()

    def _sanity_check(self):
        pass

    def __repr__(self):
        return "Points(coordinates=%s, tag='%s')" % (self.coordinate, self.tag)


class SpherePoint(Point):
    """ Point on the sphere, coordinates on latitude and longitude """

    def __init__(self, latitude, longitude, tag, weight=0.0):
        Point.__init__(self, [latitude, longitude], tag, weight=weight)

    @property
    def latitude(self):
        return self.coordinate[0]

    @property
    def longitude(self):
        return self.coordinate[1]


from .sphereweight import SphereDistRel, SphereVoronoi  # NOQA
from .sphereweight import SphereAziBin, SphereAziRel  # NOQA
