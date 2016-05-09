from spaceweight import Point, SpherePoint


def test_point():
    point = Point([1, 2, 3], "test", weight=1.0)
    assert point.dimension == (3,)


def test_spherepoint():
    point = SpherePoint(1, 2, "test", weight=1.0)
    assert point.dimension == (2,)
    assert point.latitude == 1.0
    assert point.longitude == 2.0
