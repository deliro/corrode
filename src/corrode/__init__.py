"""A Rust-like Result type for Python."""

from . import async_iterator, iterator
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
    "async_iterator",
    "do",
    "do_async",
    "is_err",
    "is_ok",
    "iterator",
]
