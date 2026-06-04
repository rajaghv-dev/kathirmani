from ingestion.windows import window_bounds


def test_10s_clip_5s_window_2s_stride():
    w = window_bounds(10, 5, 2)
    assert (0.0, 5.0) in w and (2.0, 7.0) in w and (4.0, 9.0) in w
    assert w[-1][1] == 10.0                      # covers the clip tail
    assert all(s < e for s, e in w)              # every window is valid


def test_short_clip_single_window():
    assert window_bounds(4, 5, 2) == [(0.0, 4.0)]


def test_zero_duration_empty():
    assert window_bounds(0, 5, 2) == []


def test_bad_params_raise():
    import pytest
    with pytest.raises(ValueError):
        window_bounds(10, 0, 2)
