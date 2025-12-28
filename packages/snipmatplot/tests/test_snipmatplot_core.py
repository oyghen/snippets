from snipmatplot import core


def test_discrete_cmap():
    n = 5
    rgba_list = core.discrete_cmap(n)
    assert len(rgba_list) == n
    assert all(len(rgba) == 4 for rgba in rgba_list)
    assert all(isinstance(value, float) for rgba in rgba_list for value in rgba)
