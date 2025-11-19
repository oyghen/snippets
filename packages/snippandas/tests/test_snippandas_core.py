import pandas as pd
import pytest
from pandas import DataFrame as PandasDF
from snippandas import core


class TestProfile:
    def test_default_call(self, df: PandasDF, expected_default: PandasDF):
        actual = core.profile(df)
        assert isinstance(actual, PandasDF)

        pd.testing.assert_frame_equal(actual, expected_default)
        pd.testing.assert_frame_equal(
            actual.query("mean.notnull()"),
            expected_default.loc[["b", "d"], :],
        )
        pd.testing.assert_frame_equal(
            actual.query("mean.isnull()"),
            expected_default.loc[["a", "c"], :],
        )

    def test_quantiles(self, df: PandasDF, expected_quantiles: PandasDF):
        actual = core.profile(df, q=[1, 30, 70, 99])
        assert isinstance(actual, PandasDF)

        pd.testing.assert_frame_equal(actual, expected_quantiles)

    @pytest.fixture(scope="class")
    def df(self) -> PandasDF:
        return pd.DataFrame(
            {
                "a": [True, None, False, False, True, False],
                "b": [1] * 6,
                "c": [None] * 6,
                "d": list(range(6)),
            }
        )

    @pytest.fixture(scope="class")
    def expected_default(self) -> PandasDF:
        return pd.DataFrame(
            {
                "type": {"a": "object", "b": "int64", "c": "object", "d": "int64"},
                "count": {"a": 5, "b": 6, "c": 0, "d": 6},
                "isnull": {"a": 1, "b": 0, "c": 6, "d": 0},
                "isnull_pct": {"a": 16.666666666666668, "b": 0.0, "c": 100.0, "d": 0.0},
                "unique": {"a": 2, "b": 1, "c": 0, "d": 6},
                "unique_pct": {"a": 33.333333, "b": 16.666666, "c": 0.0, "d": 100.0},
                "sum": {"a": float("nan"), "b": 6.0, "c": float("nan"), "d": 15.0},
                "mean": {"a": float("nan"), "b": 1.0, "c": float("nan"), "d": 2.5},
                "std": {"a": float("nan"), "b": 0.0, "c": float("nan"), "d": 1.8708286},
                "skewness": {"a": float("nan"), "b": 0.0, "c": float("nan"), "d": 0.0},
                "kurtosis": {"a": float("nan"), "b": 0.0, "c": float("nan"), "d": -1.2},
                "min": {"a": float("nan"), "b": 1.0, "c": float("nan"), "d": 0.0},
                "q5": {"a": float("nan"), "b": 1.0, "c": float("nan"), "d": 0.25},
                "q25": {"a": float("nan"), "b": 1.0, "c": float("nan"), "d": 1.25},
                "q50": {"a": float("nan"), "b": 1.0, "c": float("nan"), "d": 2.5},
                "q75": {"a": float("nan"), "b": 1.0, "c": float("nan"), "d": 3.75},
                "q95": {"a": float("nan"), "b": 1.0, "c": float("nan"), "d": 4.75},
                "max": {"a": float("nan"), "b": 1.0, "c": float("nan"), "d": 5.0},
            }
        )

    @pytest.fixture(scope="class")
    def expected_quantiles(self) -> PandasDF:
        return pd.DataFrame(
            {
                "type": {"a": "object", "b": "int64", "c": "object", "d": "int64"},
                "count": {"a": 5, "b": 6, "c": 0, "d": 6},
                "isnull": {"a": 1, "b": 0, "c": 6, "d": 0},
                "isnull_pct": {"a": 16.666666666666668, "b": 0.0, "c": 100.0, "d": 0.0},
                "unique": {"a": 2, "b": 1, "c": 0, "d": 6},
                "unique_pct": {"a": 33.333333, "b": 16.666666, "c": 0.0, "d": 100.0},
                "sum": {"a": float("nan"), "b": 6.0, "c": float("nan"), "d": 15.0},
                "mean": {"a": float("nan"), "b": 1.0, "c": float("nan"), "d": 2.5},
                "std": {"a": float("nan"), "b": 0.0, "c": float("nan"), "d": 1.8708286},
                "skewness": {"a": float("nan"), "b": 0.0, "c": float("nan"), "d": 0.0},
                "kurtosis": {"a": float("nan"), "b": 0.0, "c": float("nan"), "d": -1.2},
                "min": {"a": float("nan"), "b": 1.0, "c": float("nan"), "d": 0.0},
                "q1": {"a": float("nan"), "b": 1.0, "c": float("nan"), "d": 0.05},
                "q30": {"a": float("nan"), "b": 1.0, "c": float("nan"), "d": 1.5},
                "q70": {"a": float("nan"), "b": 1.0, "c": float("nan"), "d": 3.5},
                "q99": {"a": float("nan"), "b": 1.0, "c": float("nan"), "d": 4.95},
                "max": {"a": float("nan"), "b": 1.0, "c": float("nan"), "d": 5.0},
            }
        )
