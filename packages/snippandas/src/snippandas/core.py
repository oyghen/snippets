__all__ = ["profile"]

import pandas as pd
from pandas import DataFrame as PandasDF


def profile(df: PandasDF, /, *, q: list[int] | None = None) -> PandasDF:
    """Profile Pandas dataframe.

    Examples
    --------
    >>> import pandas as pd
    >>> from snippandas import core
    >>> data = {
    ...     "a": [True, None, False, False, True, False],
    ...     "b": [1] * 6,
    ...     "c": [None] * 6,
    ... }
    >>> core.profile(pd.DataFrame(data)).T
                        a          b       c
    type           object      int64  object
    count               5          6       0
    isnull              1          0       6
    isnull_pct  16.666667        0.0   100.0
    unique              2          1       0
    unique_pct  33.333333  16.666667     0.0
    sum               NaN        6.0     NaN
    mean              NaN        1.0     NaN
    std               NaN        0.0     NaN
    skewness          NaN        0.0     NaN
    kurtosis          NaN        0.0     NaN
    min               NaN        1.0     NaN
    q5                NaN        1.0     NaN
    q25               NaN        1.0     NaN
    q50               NaN        1.0     NaN
    q75               NaN        1.0     NaN
    q95               NaN        1.0     NaN
    max               NaN        1.0     NaN
    """
    num_rows, _ = df.shape
    quantiles = q or (5, 25, 50, 75, 95)

    basic_info_df = pd.concat(
        [
            df.dtypes.apply(str).to_frame("type"),
            df.count().to_frame("count"),
            (
                df.isnull()
                .sum()
                .to_frame("isnull")
                .assign(isnull_pct=lambda df: 100 * df["isnull"] / num_rows)
            ),
            (
                df.nunique()
                .to_frame("unique")
                .assign(unique_pct=lambda df: 100 * df["unique"] / num_rows)
            ),
        ],
        axis=1,
    )

    return pd.concat(
        [
            basic_info_df,
            df.sum(numeric_only=True).to_frame("sum"),
            df.mean(numeric_only=True).to_frame("mean"),
            df.std(numeric_only=True, ddof=1).to_frame("std"),
            df.skew(numeric_only=True).to_frame("skewness"),
            df.kurt(numeric_only=True).to_frame("kurtosis"),
            df.min(numeric_only=True).to_frame("min"),
            *[
                df.quantile(q / 100, numeric_only=True).to_frame(f"q{q}")
                for q in quantiles
            ],
            df.max(numeric_only=True).to_frame("max"),
        ],
        axis=1,
    )
