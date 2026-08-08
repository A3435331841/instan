# -*- coding: utf-8 -*-
"""S2 spherical state regression tests."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panotrack.geometry.bfov import BFoV
from panotrack.pipeline.state import SphericalState


def test_seam_continuity():
    st = SphericalState(BFoV(178.0, 0.0, 10.0, 10.0))
    st.update(BFoV(179.0, 0.0, 10.0, 10.0))
    st.update(BFoV(-179.0, 0.0, 10.0, 10.0))
    p = st.predict()
    assert -180.0 < p.lon <= 180.0


def test_unit_vector_interface_if_available():
    st = SphericalState(BFoV(0.0, 86.0, 8.0, 8.0))
    st.update(BFoV(45.0, 88.0, 8.0, 8.0))
    p = st.predict()
    assert np.isfinite([p.lon, p.lat]).all()


if __name__ == '__main__':
    test_seam_continuity()
    test_unit_vector_interface_if_available()
    print('ALL S2 STATE TESTS PASSED')
