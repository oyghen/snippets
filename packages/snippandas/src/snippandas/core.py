__all__ = ("display", "profile")


import numpy as np
import pandas as pd
from pandas import DataFrame as PandasDF
from tabulate import tabulate


def display(
    df: PandasDF,
    label: str | None = None,
    na_rep: str | None = "NULL",
    drop_original_index: bool = True,
    with_row_numbers: bool = True,
) -> None:  # pragma: no cover
    """Returns a stylized representation of the Pandas dataframe.

    Examples
    --------
    >>> import pandas as pd
    >>> from snippandas import core
    >>> df = pd.DataFrame([dict(x=1, y=2), dict(x=3, y=4), dict(x=None, y=6)])
    >>> core.display(df)
    +----+------+-----+
    |    |    x |   y |
    +====+======+=====+
    |  1 |    1 |   2 |
    +----+------+-----+
    |  2 |    3 |   4 |
    +----+------+-----+
    |  3 | NULL |   6 |
    +----+------+-----+
    """
    with_label = label is not None

    def print_tabulated_df() -> None:
        tabulated_df = tabulate(
            df.replace(np.nan, None),
            headers="keys",
            tablefmt="grid",
            floatfmt="_g",
            intfmt="_g",
            missingval=na_rep,
        )
        if with_label:
            n = max(map(len, tabulated_df.splitlines()))
            print(f" {label} ".center(n, fillchar="="))
        print(tabulated_df)

    if with_row_numbers:
        df = df.copy().reset_index(drop=drop_original_index)
        df.index += 1

    if get_shell_type() != "notebook":
        print_tabulated_df()
        return

    try:
        from IPython.display import display as notebook_display

        df_style = df.style.set_caption(label) if with_label else df.style
        table_styles = []
        if with_label:
            table_styles.append(
                dict(
                    selector="caption",
                    props=[("font-size", "120%"), ("font-weight", "bold")],
                )
            )

        # Function to apply 'g' formatting
        def general_format(x) -> str:
            if pd.isnull(x):
                return na_rep
            return f"{x:_g}"

        df_style = (
            df_style.format(
                na_rep=na_rep,
                formatter={
                    col: general_format
                    for col in df.select_dtypes(include=[np.number]).columns
                },
            )
            .highlight_null(props="color: lightgray; background-color: transparent")
            .set_table_styles(table_styles)
        )

        notebook_display(df_style)

    except (ModuleNotFoundError, ImportError):
        print_tabulated_df()


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


def get_shell_type() -> str:
    """Returns the type of the current shell session.

    The function identifies whether code is executed from within
    a 'python', 'ipython', or 'notebook' session.
    """
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            return "python"

        shell = get_ipython().__class__.__name__

        if shell == "TerminalInteractiveShell":
            return "ipython"

        elif shell == "ZMQInteractiveShell":
            return "notebook"

        else:
            return "python"

    except (ModuleNotFoundError, ImportError, NameError):
        return "python"
