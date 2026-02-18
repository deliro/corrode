"""A Rust-like Result type for Python."""

from .async_iterator import (
    collect_async_unordered,
    filter_err_unordered,
    filter_ok_unordered,
    map_collect_async_unordered,
    partition_async_unordered,
)
from .iterator import (
    collect,
    filter_err,
    filter_ok,
    map_collect,
    partition,
    try_reduce,
)
from .result import (
    Err,
    Ok,
    Result,
    UnwrapError,
    as_async_result,
    as_result,
    do,
    do_async,
    is_err,
    is_ok,
)

__all__ = [
    "Err",
    "Ok",
    "Result",
    "UnwrapError",
    "as_async_result",
    "as_result",
    "collect",
    "collect_async_unordered",
    "do",
    "do_async",
    "filter_err",
    "filter_err_unordered",
    "filter_ok",
    "filter_ok_unordered",
    "is_err",
    "is_ok",
    "map_collect",
    "map_collect_async_unordered",
    "partition",
    "partition_async_unordered",
    "try_reduce",
]
