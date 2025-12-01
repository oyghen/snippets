import random

import numpy as np
import numpy.testing as npt
import pytest
from snipnumpy import core


@pytest.mark.parametrize(
    "seed",
    [
        None,
        0,
        1.0,
        random.Random(),
        np.random.RandomState(),
        np.random.default_rng(),
        "invalid seed",
    ],
)
def test_check_random_state(seed):
    valid_types = (
        int,
        float,
        random.Random,
        np.random.RandomState,
        np.random.Generator,
    )
    if seed is None or isinstance(seed, valid_types):
        assert isinstance(core.check_random_state(seed), np.random.Generator)
    else:
        with pytest.raises(ValueError):
            core.check_random_state(seed)


def test_check_vector():
    x = [1, 2, 3]
    actual = core.check_vector(x)
    npt.assert_array_equal(actual, np.array(x))

    with pytest.raises(
        TypeError,
        match=r"input must be a vector-like object - it has shape=\(1, 2\)",
    ):
        core.check_vector([[1, 2]])

    with pytest.raises(
        TypeError,
        match=r"x with n=2 - n must be an integer in \[3, inf\)",
    ):
        core.check_vector([1, 2], n_min=3)

    with pytest.raises(
        TypeError,
        match=r"x with n=1 - n must be an integer in \[2, 2\]",
    ):
        core.check_vector([1], n_min=2, n_max=2)


@pytest.mark.parametrize(
    "data, pos_label, expected",
    [
        ([0, 1, 2, 1, 0, 1], 1, np.array([False, True, False, True, False, True])),
        (
            ["cat", "dog", "cat", "bird", "cat", "dog"],
            "dog",
            np.array([False, True, False, False, False, True]),
        ),
    ],
)
def test_create_boolean_array(
    data: np.ndarray,
    pos_label: int | str,
    expected: np.ndarray,
):
    actual = core.create_boolean_array(data, pos_label)
    npt.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        ([], np.nan),
        ([np.nan], np.nan),
        ([np.nan, np.nan], np.nan),
        ([1], np.nan),
        ([1, 1], 0.0),
        ([98, 127, 82, 67, 121, np.nan, 119, 92, 110, 113, 107], 5.963220606350229),
    ],
)
def test_stderr(x: core.ArrayLike, expected: float):
    actual = core.stderr(x)
    npt.assert_array_equal(actual, expected)
