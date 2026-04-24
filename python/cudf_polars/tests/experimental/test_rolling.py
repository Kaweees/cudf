# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_rolling_datetime(streaming_engine):
    df = pl.LazyFrame(
        {
            "dt": pl.Series(
                [
                    datetime(2020, 1, 1, 13, 45, 48),
                    datetime(2020, 1, 1, 16, 42, 13),
                    datetime(2020, 1, 1, 16, 45, 9),
                    datetime(2020, 1, 2, 18, 12, 48),
                    datetime(2020, 1, 3, 19, 45, 32),
                    datetime(2020, 1, 8, 23, 16, 43),
                ],
                dtype=pl.Datetime("ns"),
            ),
            "a": [3, 7, 5, 9, 2, 1],
        }
    )
    q = df.rolling("dt", period="2d").agg(sum_a=pl.sum("a"))
    assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.parametrize(
    "streaming_engine",
    [{"executor_options": {"max_rows_per_partition": 2}}],
    indirect=True,
)
@pytest.mark.parametrize("closed", ["left", "right", "both", "none"])
def test_rolling_integer_period(streaming_engine, closed) -> None:
    df = pl.LazyFrame(
        {
            "orderby": [1, 4, 8, 10, 12, 13, 14, 22],
            "values": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    q = df.rolling("orderby", period="4i", closed=closed).agg(
        sum_values=pl.col("values").sum(),
        min_values=pl.col("values").min(),
        max_values=pl.col("values").max(),
        count=pl.len(),
    )

    assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.parametrize(
    "streaming_engine",
    [{"executor_options": {"max_rows_per_partition": 2}}],
    indirect=True,
)
@pytest.mark.parametrize(
    "period, offset",
    [
        ("10i", "-5i"),
        ("10i", "20i"),
        ("10i", "-30i"),
    ],
    ids=["nonzero-overlap", "fully-leading", "fully-trailing"],
)
@pytest.mark.parametrize("closed", ["left", "right", "both", "none"])
def test_rolling_integer_offset(streaming_engine, period, offset, closed) -> None:
    df = pl.LazyFrame(
        {
            "orderby": [0, 5, 10, 15, 20, 25, 30, 35],
            "values": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    q = df.rolling("orderby", period=period, offset=offset, closed=closed).agg(
        sum_values=pl.col("values").sum(),
        count=pl.len(),
    )

    assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.parametrize(
    "streaming_engine",
    [{"executor_options": {"max_rows_per_partition": 2}}],
    indirect=True,
)
@pytest.mark.parametrize("closed", ["left", "right", "both", "none"])
def test_rolling_datetime_period(streaming_engine, closed) -> None:
    df = pl.LazyFrame(
        {
            "dt": pl.Series(
                [
                    datetime(2020, 1, 1, 13, 45, 48),
                    datetime(2020, 1, 1, 16, 42, 13),
                    datetime(2020, 1, 1, 16, 45, 9),
                    datetime(2020, 1, 2, 18, 12, 48),
                    datetime(2020, 1, 3, 19, 45, 32),
                    datetime(2020, 1, 8, 23, 16, 43),
                ],
                dtype=pl.Datetime("us"),
            ),
            "values": [3, 7, 5, 9, 2, 1],
        }
    )
    q = df.rolling("dt", period="2d", closed=closed).agg(
        sum_values=pl.col("values").sum()
    )

    assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.parametrize(
    "streaming_engine",
    [{"executor_options": {"max_rows_per_partition": 2}}],
    indirect=True,
)
@pytest.mark.parametrize("closed", ["left", "right", "both", "none"])
def test_grouped_rolling_warns(streaming_engine, closed) -> None:
    df = pl.LazyFrame(
        {
            "key": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "orderby": [1, 4, 8, 10, 1, 3, 6, 10],
            "values": [1, 2, 3, 4, 10, 20, 30, 40],
        }
    )
    q = df.rolling("orderby", period="4i", closed=closed, group_by="key").agg(
        sum_values=pl.col("values").sum(),
        count=pl.len(),
    )

    with pytest.warns(
        UserWarning, match="Rolling does not support multiple partitions"
    ):
        assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.parametrize(
    "streaming_engine",
    [{"executor_options": {"max_rows_per_partition": 1}}],
    indirect=True,
)
def test_over_in_filter_unsupported(streaming_engine) -> None:
    q = pl.concat(
        [
            pl.LazyFrame({"k": ["x", "y"], "v": [3, 2]}),
            pl.LazyFrame({"k": ["x", "y"], "v": [5, 7]}),
        ]
    ).filter(pl.len().over("k") == 2)

    with pytest.warns(
        UserWarning,
        match=r"over\(...\) inside filter is not supported for multiple partitions.*",
    ):
        assert_gpu_result_equal(q, engine=streaming_engine)
