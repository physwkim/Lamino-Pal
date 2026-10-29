from lamino_pal import units
from lamino_pal.project import Scan


def test_bin2_504_is_full_1008_5():
    s = Scan(tag="s", kind="step", h5="x.h5", bin=2)
    assert units.center_for_scan(s, 1008.5) == 504.0
    assert units.center_full(s, 504.0) == 1008.5
    one = s.model_copy(update={"bin": 1})
    assert units.center_for_scan(one, 1008.5) == units.center_full(one, 1008.5) == 1008.5
