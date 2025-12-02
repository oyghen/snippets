__all__ = (
    "add_prefix",
    "add_suffix",
    "all_col",
    "any_col",
    "assert_dataframe_equal",
    "assert_row_count_equal",
    "assert_row_value_equal",
    "assert_schema_equal",
    "bool_to_int",
    "bool_to_str",
    "check_column_present",
    "count_nulls",
    "cvf",
    "date_range",
    "filter_date",
    "has_column",
    "is_dataframe_equal",
    "is_row_count_equal",
    "is_row_value_equal",
    "is_schema_equal",
    "join",
    "peek",
    "str_to_col",
    "union",
    "with_date_diff_ago",
    "with_date_diff_ahead",
    "with_endofweek_date",
    "with_increasing_id",
    "with_index",
    "with_startofweek_date",
    "with_weekday",
)

import datetime as dt
import functools
import math
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import snippandas
import toolz
from pyspark.sql import Column as SparkCol
from pyspark.sql import DataFrame as SparkDF
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T
from toolz import curried


def add_prefix(df: SparkDF, prefix: str, subset: list[str] | None = None) -> SparkDF:
    """Add prefix to column names.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame([Row(x=1, y=2)])
    >>> core.add_prefix(df, "pfx_").show()
    +-----+-----+
    |pfx_x|pfx_y|
    +-----+-----+
    |    1|    2|
    +-----+-----+
    <BLANKLINE>
    """
    cols = subset or df.columns
    for col in cols:
        df = df.withColumnRenamed(col, f"{prefix}{col}")
    return df


def add_suffix(df: SparkDF, suffix: str, subset: list[str] | None = None) -> SparkDF:
    """Add suffix to column names.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame([Row(x=1, y=2)])
    >>> core.add_suffix(df, "_sfx").show()
    +-----+-----+
    |x_sfx|y_sfx|
    +-----+-----+
    |    1|    2|
    +-----+-----+
    <BLANKLINE>
    """
    cols = subset or df.columns
    for col in cols:
        df = df.withColumnRenamed(col, f"{col}{suffix}")
    return df


def all_col(*cols: str | Iterable[str]) -> SparkCol:
    """Evaluate if all columns are true.

    Notes
    -----
    A NULL value is considered false.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(x=True, y=True),
    ...         Row(x=True, y=False),
    ...         Row(x=True, y=None),
    ...         Row(x=None, y=False),
    ...         Row(x=None, y=None),
    ...     ]
    ... )
    >>> df.withColumn("all_cols_true", core.all_col("x", "y")).show()
    +----+-----+-------------+
    |   x|    y|all_cols_true|
    +----+-----+-------------+
    |true| true|         true|
    |true|false|        false|
    |true| NULL|        false|
    |NULL|false|        false|
    |NULL| NULL|        false|
    +----+-----+-------------+
    <BLANKLINE>
    """
    return functools.reduce(
        SparkCol.__and__,
        toolz.pipe(
            cols,
            flatten,
            curried.map(lambda col: F.coalesce(str_to_col(col), F.lit(False))),
        ),
    )


def any_col(*cols: str | Iterable[str]) -> SparkCol:
    """Evaluate if any column is true.

    Notes
    -----
    A NULL value is considered false.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(x=True, y=True),
    ...         Row(x=True, y=False),
    ...         Row(x=True, y=None),
    ...         Row(x=None, y=False),
    ...         Row(x=None, y=None),
    ...     ]
    ... )
    >>> df.withColumn("any_col_true", core.any_col("x", "y")).show()
    +----+-----+------------+
    |   x|    y|any_col_true|
    +----+-----+------------+
    |true| true|        true|
    |true|false|        true|
    |true| NULL|        true|
    |NULL|false|       false|
    |NULL| NULL|       false|
    +----+-----+------------+
    <BLANKLINE>
    """
    return functools.reduce(
        SparkCol.__or__,
        toolz.pipe(
            cols,
            flatten,
            curried.map(lambda col: F.coalesce(str_to_col(col), F.lit(False))),
        ),
    )


def assert_dataframe_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> None:
    """Validate both dataframes are equal."""
    assert_schema_equal(lft_df, rgt_df)
    assert_row_count_equal(lft_df, rgt_df)
    assert_row_value_equal(lft_df, rgt_df)


def assert_row_count_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> None:
    """Validate row counts of both dataframes are equal."""
    num_lft = lft_df.count()
    num_rgt = rgt_df.count()

    if num_lft != num_rgt:
        raise ValueError(f"{num_lft=}, {num_rgt=}")


def assert_row_value_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> None:
    """Validate row values of both dataframes are equal."""
    lft_rows = lft_df.subtract(rgt_df)
    rgt_rows = rgt_df.subtract(lft_df)

    num_lft = lft_rows.count()
    num_rgt = rgt_rows.count()

    is_equal = (num_lft == 0) and (num_rgt == 0)

    if not is_equal:
        raise ValueError(f"{lft_rows=}, {rgt_rows=}, {num_lft=}, {num_rgt=}")


def assert_schema_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> None:
    """Validate schemas of both dataframes are equal."""
    # only check column name and type - ignore nullable property
    lft_schema = lft_df.schema.simpleString()
    rgt_schema = rgt_df.schema.simpleString()

    if lft_schema != rgt_schema:
        raise ValueError(f"{lft_schema=}, {rgt_schema=}")


def bool_to_int(df: SparkDF, subset: list[str] | None = None) -> SparkDF:
    """Cast values of Boolean columns to 0/1 integer values.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(x=True, y=False, z=None),
    ...         Row(x=False, y=None, z=True),
    ...         Row(x=True, y=None, z=None),
    ...     ]
    ... )
    >>> core.bool_to_int(df).show()
    +---+----+----+
    |  x|   y|   z|
    +---+----+----+
    |  1|   0|NULL|
    |  0|NULL|   1|
    |  1|NULL|NULL|
    +---+----+----+
    <BLANKLINE>

    >>> core.bool_to_int(df, subset=["y", "z"]).show()
    +-----+----+----+
    |    x|   y|   z|
    +-----+----+----+
    | true|   0|NULL|
    |false|NULL|   1|
    | true|NULL|NULL|
    +-----+----+----+
    <BLANKLINE>
    """
    cols = subset or df.columns
    bool_cols = [
        field
        for field in df.schema.fields
        if "boolean" in field.dataType.simpleString() and field.name in cols
    ]
    for bool_col in bool_cols:
        df = df.withColumn(bool_col, F.col(bool_col).cast(T.IntegerType()))
    return df


def bool_to_str(df: SparkDF, subset: list[str] | None = None) -> SparkDF:
    """Cast values of Boolean columns to string values.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(x=True, y=False, z=None),
    ...         Row(x=False, y=None, z=True),
    ...         Row(x=True, y=None, z=None),
    ...     ]
    ... )
    >>> core.bool_to_str(df).show()
    +-----+-----+----+
    |    x|    y|   z|
    +-----+-----+----+
    | true|false|NULL|
    |false| NULL|true|
    | true| NULL|NULL|
    +-----+-----+----+
    <BLANKLINE>

    >>> core.bool_to_str(df, subset=["y", "z"]).printSchema()
    root
     |-- x: boolean (nullable = true)
     |-- y: string (nullable = true)
     |-- z: string (nullable = true)
    <BLANKLINE>
    """
    cols = subset or df.columns
    bool_cols = [
        field
        for field in df.schema.fields
        if "boolean" in field.dataType.simpleString() and field.name in cols
    ]
    for bool_col in bool_cols:
        df = df.withColumn(bool_col, F.col(bool_col).cast(T.StringType()))
    return df


def check_column_present(df: SparkDF, *cols: str | Iterable[str]) -> SparkDF:
    """Check if columns are present in dataframe."""
    missing_cols = [col for col in flatten(cols) if col not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(missing_cols)
    return df


def count_nulls(df: SparkDF, subset: list[str] | None = None) -> SparkDF:
    """Count NULL values in Spark dataframe.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(x=1, y=2, z=None),
    ...         Row(x=4, y=None, z=6),
    ...         Row(x=10, y=None, z=None),
    ...     ]
    ... )
    >>> core.count_nulls(df).show()
    +---+---+---+
    |  x|  y|  z|
    +---+---+---+
    |  0|  2|  2|
    +---+---+---+
    <BLANKLINE>
    """
    cols = subset or df.columns
    return df.agg(*[F.sum(F.isnull(c).cast(T.LongType())).alias(c) for c in cols])


def cvf(df: SparkDF, *cols: str | SparkCol | Iterable[str | SparkCol]) -> SparkDF:
    """Count value frequency.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(x="a"),
    ...         Row(x="c"),
    ...         Row(x="b"),
    ...         Row(x="g"),
    ...         Row(x="h"),
    ...         Row(x="a"),
    ...         Row(x="g"),
    ...         Row(x="a"),
    ...     ]
    ... )
    >>> core.cvf(df, "x").show()
    +---+-----+-------+-----------+-------------+
    |  x|count|percent|cumul_count|cumul_percent|
    +---+-----+-------+-----------+-------------+
    |  a|    3|   37.5|          3|         37.5|
    |  g|    2|   25.0|          5|         62.5|
    |  b|    1|   12.5|          6|         75.0|
    |  c|    1|   12.5|          7|         87.5|
    |  h|    1|   12.5|          8|        100.0|
    +---+-----+-------+-----------+-------------+
    <BLANKLINE>
    """
    columns = toolz.pipe(cols, flatten, curried.map(str_to_col), list)
    w0 = Window.partitionBy(F.lit(True))
    w1 = w0.orderBy(F.desc("count"), *columns)
    return (
        df.groupby(columns)
        .count()
        .withColumn("percent", 100 * F.col("count") / F.sum("count").over(w0))
        .withColumn("cumul_count", F.sum("count").over(w1))
        .withColumn("cumul_percent", F.sum("percent").over(w1))
        .orderBy("cumul_count")
    )


def date_range(
    df: SparkDF,
    min_date: str | dt.date,
    max_date: str | dt.date,
    id_col: str,
    new_col: str,
) -> SparkDF:
    """Generate sequence of consecutive dates between two dates for each distinct ID.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(id=1),
    ...         Row(id=1),
    ...         Row(id=3),
    ...         Row(id=2),
    ...         Row(id=2),
    ...         Row(id=3),
    ...     ]
    ... )
    >>> (
    ...     core.date_range(df, "2023-05-01", "2023-05-03", "id", "d")
    ...     .orderBy("id", "d")
    ...     .show()
    ... )
    +---+----------+
    | id|         d|
    +---+----------+
    |  1|2023-05-01|
    |  1|2023-05-02|
    |  1|2023-05-03|
    |  2|2023-05-01|
    |  2|2023-05-02|
    |  2|2023-05-03|
    |  3|2023-05-01|
    |  3|2023-05-02|
    |  3|2023-05-03|
    +---+----------+
    <BLANKLINE>
    """
    return (
        df.select(id_col)
        .distinct()
        .withColumn("min_date", F.to_date(F.lit(min_date), "yyyy-MM-dd"))
        .withColumn("max_date", F.to_date(F.lit(max_date), "yyyy-MM-dd"))
        .select(
            id_col,
            F.expr("sequence(min_date, max_date, interval 1 day)").alias(new_col),
        )
        .withColumn(new_col, F.explode(new_col))
    )


def filter_date(
    df: SparkDF,
    date_col: str,
    ref_date: str | dt.date,
    num_days: int | float,
) -> SparkDF:
    """Filter dataframe such that `num_days` corresponds to the number of distinct dates
    when consecutively counting backwards from reference date.

    Notes
    -----
    - reference date is inclusive
    - relative date = date_ago(ref_date, num_days) is exclusive
    - If `num_days=float("inf")`, returns all dates prior to reference date (inclusive)

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(d="2024-01-01"),
    ...         Row(d="2024-01-02"),
    ...         Row(d="2024-01-03"),
    ...         Row(d="2024-01-04"),
    ...         Row(d="2024-01-05"),
    ...         Row(d="2024-01-06"),
    ...         Row(d="2024-01-07"),
    ...         Row(d="2024-01-08"),
    ...     ],
    ... )
    >>> core.filter_date(df, "d", ref_date="2024-01-07", num_days=3).show()
    +----------+
    |         d|
    +----------+
    |2024-01-05|
    |2024-01-06|
    |2024-01-07|
    +----------+
    <BLANKLINE>

    >>> core.filter_date(df, "d", ref_date="2024-01-07", num_days=float("inf")).show()
    +----------+
    |         d|
    +----------+
    |2024-01-01|
    |2024-01-02|
    |2024-01-03|
    |2024-01-04|
    |2024-01-05|
    |2024-01-06|
    |2024-01-07|
    +----------+
    <BLANKLINE>
    """
    if not isinstance(num_days, (int, float)):
        raise TypeError(f"{type(num_days)=} - must be an int or float")

    if isinstance(num_days, int) and num_days < 1:
        raise ValueError(f"{num_days=} - must be a positive integer")

    if isinstance(num_days, float) and not math.isinf(num_days):
        raise ValueError(f'{num_days=} - only valid float value: float("inf")')

    date_diff_ago = "_date_diff_ago_"

    return (
        with_date_diff_ago(df, date_col, ref_date, new_col=date_diff_ago)
        .where((F.col(date_diff_ago) >= 0) & (F.col(date_diff_ago) < num_days))
        .drop(date_diff_ago)
    )


def has_column(df: SparkDF, *cols: str | Iterable[str]) -> bool:
    """Evaluate if all columns are present in dataframe."""
    try:
        check_column_present(df, cols)
        return True
    except ValueError:
        return False


def is_dataframe_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> bool:
    """Evaluate if both dataframes are equal."""
    try:
        assert_schema_equal(lft_df, rgt_df)
        assert_row_count_equal(lft_df, rgt_df)
        assert_row_value_equal(lft_df, rgt_df)
        return True
    except Exception:
        return False


def is_row_count_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> bool:
    """Evaluate if row counts of both dataframes are equal."""
    try:
        assert_row_count_equal(lft_df, rgt_df)
        return True
    except ValueError:
        return False


def is_row_value_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> bool:
    """Evaluate if rows of both dataframes are equal."""
    try:
        assert_row_value_equal(lft_df, rgt_df)
        return True
    except ValueError:
        return False


def is_schema_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> bool:
    """Evaluate if schemas of both dataframes are equal."""
    try:
        assert_schema_equal(lft_df, rgt_df)
        return True
    except ValueError:
        return False


def join(
    *dataframes: SparkDF | Iterable[SparkDF],
    on: str | list[str],
    how: str = "inner",
) -> SparkDF:
    """Join iterable of Spark dataframes.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df1 = spark.createDataFrame([Row(id=1, x="a"), Row(id=2, x="b")])
    >>> df2 = spark.createDataFrame([Row(id=1, y="c"), Row(id=2, y="d")])
    >>> df3 = spark.createDataFrame([Row(id=1, z="e"), Row(id=2, z="f")])
    >>> core.join(df1, df2, df3, on="id").show()
    +---+---+---+---+
    | id|  x|  y|  z|
    +---+---+---+---+
    |  1|  a|  c|  e|
    |  2|  b|  d|  f|
    +---+---+---+---+
    <BLANKLINE>
    """
    return functools.reduce(
        functools.partial(SparkDF.join, on=on, how=how),
        flatten(dataframes),
    )


def peek(
    df: SparkDF,
    n: int = 6,
    shape: bool = False,
    cache: bool = False,
    schema: bool = False,
    label: str | None = None,
) -> SparkDF:
    """Peek at dataframe between transformations.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(x=1, y="a"),
    ...         Row(x=3, y=None),
    ...         Row(x=None, y="c"),
    ...     ]
    ... )
    >>> df.show()
    +----+----+
    |   x|   y|
    +----+----+
    |   1|   a|
    |   3|NULL|
    |NULL|   c|
    +----+----+
    <BLANKLINE>
    >>> filtered_df = (
    ...     core.peek(df, shape=True)
    ...     .where("x IS NOT NULL")
    ...     .transform(lambda df: core.peek(df, shape=True))
    ... )
    shape=(3, 2)
    +----+------+------+
    |    |    x | y    |
    +====+======+======+
    |  1 |    1 | a    |
    +----+------+------+
    |  2 |    3 | NULL |
    +----+------+------+
    |  3 | NULL | c    |
    +----+------+------+
    shape=(2, 2)
    +----+-----+------+
    |    |   x | y    |
    +====+=====+======+
    |  1 |   1 | a    |
    +----+-----+------+
    |  2 |   3 | NULL |
    +----+-----+------+
    """
    df = df if df.is_cached else df.cache() if cache else df

    if schema:
        df.printSchema()

    if shape:
        num_rows = num_to_str(df.count())
        num_cols = num_to_str(len(df.columns))
        print(f"shape=({num_rows}, {num_cols})")

    if n > 0:
        snippandas.core.display(
            df=df.limit(n).transform(lambda df: bool_to_str(df)).toPandas(),
            label=label,
        )

    return df


def str_to_col(x: str | SparkCol, /) -> SparkCol:
    """Cast string ``x`` to Spark column else return ``x``.

    Examples
    --------
    >>> from pyspark.sql import functions as F
    >>> from snipspark import core
    >>> core.str_to_col("x")
    Column<'x'>

    >>> core.str_to_col(F.col("x"))
    Column<'x'>
    """
    return F.col(x) if isinstance(x, str) else x


def union(*dataframes: SparkDF | Iterable[SparkDF]) -> SparkDF:
    """Union iterable of Spark dataframes by name.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df1 = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> df2 = spark.createDataFrame([Row(x=5, y=6), Row(x=7, y=8)])
    >>> df3 = spark.createDataFrame([Row(x=0, y=1), Row(x=2, y=3)])
    >>> core.union(df1, df2, df3).show()
    +---+---+
    |  x|  y|
    +---+---+
    |  1|  2|
    |  3|  4|
    |  5|  6|
    |  7|  8|
    |  0|  1|
    |  2|  3|
    +---+---+
    <BLANKLINE>
    """
    # noinspection PyTypeChecker
    return functools.reduce(SparkDF.unionByName, flatten(dataframes))


def with_date_diff_ago(
    df: SparkDF,
    date_col: str,
    ref_date: str | dt.date,
    new_col: str,
) -> SparkDF:
    """Add a column showing date differences with respect to the reference date,
    where differences to past dates are positive integers.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(d="2024-01-01"),
    ...         Row(d="2024-01-02"),
    ...         Row(d="2024-01-03"),
    ...         Row(d="2024-01-04"),
    ...         Row(d="2024-01-05"),
    ...         Row(d="2024-01-06"),
    ...         Row(d="2024-01-07"),
    ...         Row(d="2024-01-08"),
    ...         Row(d="2024-01-09"),
    ...     ],
    ... )
    >>> core.with_date_diff_ago(df, "d", "2024-01-07", "diff").show()
    +----------+----+
    |         d|diff|
    +----------+----+
    |2024-01-01|   6|
    |2024-01-02|   5|
    |2024-01-03|   4|
    |2024-01-04|   3|
    |2024-01-05|   2|
    |2024-01-06|   1|
    |2024-01-07|   0|
    |2024-01-08|  -1|
    |2024-01-09|  -2|
    +----------+----+
    <BLANKLINE>
    """
    return df.withColumn(new_col, F.datediff(F.lit(ref_date), str_to_col(date_col)))


def with_date_diff_ahead(
    df: SparkDF,
    date_col: str,
    ref_date: str | dt.date,
    new_col: str,
) -> SparkDF:
    """Add a column showing date differences with respect to the reference date,
    where differences to future dates are positive integers.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(d="2024-01-01"),
    ...         Row(d="2024-01-02"),
    ...         Row(d="2024-01-03"),
    ...         Row(d="2024-01-04"),
    ...         Row(d="2024-01-05"),
    ...         Row(d="2024-01-06"),
    ...         Row(d="2024-01-07"),
    ...         Row(d="2024-01-08"),
    ...         Row(d="2024-01-09"),
    ...     ],
    ... )
    >>> core.with_date_diff_ahead(df, "d", "2024-01-07", "diff").show()
    +----------+----+
    |         d|diff|
    +----------+----+
    |2024-01-01|  -6|
    |2024-01-02|  -5|
    |2024-01-03|  -4|
    |2024-01-04|  -3|
    |2024-01-05|  -2|
    |2024-01-06|  -1|
    |2024-01-07|   0|
    |2024-01-08|   1|
    |2024-01-09|   2|
    +----------+----+
    <BLANKLINE>
    """
    return df.withColumn(new_col, F.datediff(str_to_col(date_col), F.lit(ref_date)))


def with_endofweek_date(
    df: SparkDF,
    date_col: str,
    new_col: str,
    last_weekday: str = "Sun",
) -> SparkDF:
    """Add column with the end of the week date.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(d="2023-05-01"),
    ...         Row(d=None),
    ...         Row(d="2023-05-03"),
    ...         Row(d="2023-05-08"),
    ...         Row(d="2023-05-21"),
    ...     ],
    ... )
    >>> core.with_endofweek_date(df, "d", "endofweek").show()
    +----------+----------+
    |         d| endofweek|
    +----------+----------+
    |2023-05-01|2023-05-07|
    |      NULL|      NULL|
    |2023-05-03|2023-05-07|
    |2023-05-08|2023-05-14|
    |2023-05-21|2023-05-21|
    +----------+----------+
    <BLANKLINE>

    >>> core.with_endofweek_date(df, "d", "endofweek", "Sat").show()
    +----------+----------+
    |         d| endofweek|
    +----------+----------+
    |2023-05-01|2023-05-06|
    |      NULL|      NULL|
    |2023-05-03|2023-05-06|
    |2023-05-08|2023-05-13|
    |2023-05-21|2023-05-27|
    +----------+----------+
    <BLANKLINE>
    """
    tmp_col = "_weekday_"
    return (
        with_weekday(df, date_col, tmp_col)
        .withColumn(
            new_col,
            F.when(F.col(tmp_col).isNull(), None)
            .when(F.col(tmp_col) == last_weekday, F.col(date_col))
            .otherwise(F.next_day(F.col(date_col), last_weekday)),
        )
        .drop(tmp_col)
    )


def with_increasing_id(df: SparkDF, new_col: str, /) -> SparkDF:
    """Add column with monotonically increasing id.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame([Row(x="a"), Row(x="b"), Row(x="c"), Row(x="d")])
    >>> core.with_increasing_id(df, "id").show()  # doctest: +SKIP
    +---+-----------+
    |  x|         id|
    +---+-----------+
    |  a| 8589934591|
    |  b|25769803776|
    |  c|42949672960|
    |  d|60129542144|
    +---+-----------+
    <BLANKLINE>
    """
    return df.withColumn(new_col, F.monotonically_increasing_id())  # pragma: no cover


def with_index(df: SparkDF, new_col: str) -> SparkDF:
    """Add column with an index of consecutive positive integers.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame([Row(x="a"), Row(x="b"), Row(x="c"), Row(x="d")])
    >>> core.with_index(df, "idx").show()
    +---+---+
    |  x|idx|
    +---+---+
    |  a|  1|
    |  b|  2|
    |  c|  3|
    |  d|  4|
    +---+---+
    <BLANKLINE>
    """
    w = Window.partitionBy(F.lit(1)).orderBy(F.monotonically_increasing_id())
    return df.withColumn(new_col, F.row_number().over(w))


def with_startofweek_date(
    df: SparkDF,
    date_col: str,
    new_col: str,
    last_weekday: str = "Sun",
) -> SparkDF:
    """Add column with the start of the week date.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(d="2023-05-01"),
    ...         Row(d=None),
    ...         Row(d="2023-05-03"),
    ...         Row(d="2023-05-08"),
    ...         Row(d="2023-05-21"),
    ...     ],
    ... )
    >>> core.with_startofweek_date(df, "d", "startofweek").show()
    +----------+-----------+
    |         d|startofweek|
    +----------+-----------+
    |2023-05-01| 2023-05-01|
    |      NULL|       NULL|
    |2023-05-03| 2023-05-01|
    |2023-05-08| 2023-05-08|
    |2023-05-21| 2023-05-15|
    +----------+-----------+
    <BLANKLINE>

    >>> core.with_startofweek_date(df, "d", "startofweek", "Sat").show()
    +----------+-----------+
    |         d|startofweek|
    +----------+-----------+
    |2023-05-01| 2023-04-30|
    |      NULL|       NULL|
    |2023-05-03| 2023-04-30|
    |2023-05-08| 2023-05-07|
    |2023-05-21| 2023-05-21|
    +----------+-----------+
    <BLANKLINE>
    """
    tmp_col = "_endofweek_"
    return (
        with_endofweek_date(df, date_col, tmp_col, last_weekday)
        .withColumn(new_col, F.date_sub(tmp_col, 6))
        .drop(tmp_col)
    )


def with_weekday(df: SparkDF, date_col: str, new_col: str) -> SparkDF:
    """Add column with the name of the weekday.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from snipspark import core
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [Row(d="2023-05-01"), Row(d=None), Row(d="2023-05-03")]
    ... )
    >>> core.with_weekday(df, "d", "weekday").show()
    +----------+-------+
    |         d|weekday|
    +----------+-------+
    |2023-05-01|    Mon|
    |      NULL|   NULL|
    |2023-05-03|    Wed|
    +----------+-------+
    <BLANKLINE>
    """

    def determine_weekday(date_col: str, /) -> SparkCol:
        weekday_int = F.dayofweek(date_col)
        return (
            F.when(weekday_int == 1, "Sun")
            .when(weekday_int == 2, "Mon")
            .when(weekday_int == 3, "Tue")
            .when(weekday_int == 4, "Wed")
            .when(weekday_int == 5, "Thu")
            .when(weekday_int == 6, "Fri")
            .when(weekday_int == 7, "Sat")
            .otherwise(None)
        )

    return df.withColumn(new_col, determine_weekday(date_col))


def flatten(*items: Any | Iterable[Any]):
    """Flatten iterable of items."""

    def _flatten(items):
        for item in items:
            if isinstance(item, (Iterator, Sequence)) and not isinstance(item, str):
                yield from _flatten(item)
            else:
                yield item

    return _flatten(items)


def num_to_str(x: int | float, /) -> str:
    """Cast number to string with underscores as thousands separator."""
    f, i = math.modf(x)
    integer = f"{int(i):_}"
    fractional = f"{f:g}".lstrip("0")
    return integer if math.isclose(f, 0) else f"{integer}{fractional}"
