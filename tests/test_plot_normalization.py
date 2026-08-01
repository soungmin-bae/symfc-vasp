from matplotlib.colors import TwoSlopeNorm


def test_asymmetric_gruneisen_range_keeps_zero_at_colormap_center():
    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=20)
    assert norm(-5) < 0.5
    assert norm(0) == 0.5
    assert norm(5) > 0.5

