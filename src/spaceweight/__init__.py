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
import collections
import numpy as np

__version__ = "0.1.4"

# Setup the logger.
logger = logging.getLogger("spaceweighting")
logger.setLevel(logging.DEBUG)
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

    def __init__(self, coordinate, tag, weight=0.0):
        self.coordinate = np.array(coordinate)
        self.tag = tag
        self.weight = weight

        self._sanity_check()

    def _sanity_check(self):
        pass

    def __str__(self):
        return "Points(locations=%s, tag='%s')" % (self.coordinate, self.tag)
