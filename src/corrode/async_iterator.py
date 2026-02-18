"""Async iterator utilities for Result."""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable, Iterator
from typing import TypeVar

from .result import Err, Ok, Result

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")

_R = TypeVar("_R")
_CoroOrTask = Coroutine[object, object, _R] | asyncio.Task[_R]


async def _cancel_all(
    pending: set[asyncio.Task[_R]],
    it: Iterator[_CoroOrTask[_R]],
) -> None:
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for item in it:
        if isinstance(item, asyncio.Task):
            item.cancel()
        else:
            item.close()


def _make_pending(
    it: Iterator[_CoroOrTask[_R]],
    concurrency: int | None,
) -> set[asyncio.Task[_R]]:
    return {
        asyncio.ensure_future(item)
        for item in (it if concurrency is None else itertools.islice(it, concurrency))
    }


async def collect_unordered(
    iterable: Iterable[_CoroOrTask[Result[T, E]]],
    *,
    concurrency: int | None = None,
) -> Result[list[T], E]:
    """
    Await an iterable of coroutines or tasks concurrently, collecting results into ``Ok[list]``.

    Results are returned in completion order, not input order.
    Returns the first ``Err`` encountered, cancelling remaining tasks.

    *concurrency* limits how many run at the same time.
    ``None`` means unlimited — all are scheduled at once.

    **Exceptions**: if a coroutine raises, the exception propagates and all remaining
    tasks are cancelled. If multiple tasks raise in the same ``asyncio.wait`` batch,
    only one exception propagates — the rest are silently discarded.

    Example::

        collect_unordered([fetch(1), fetch(2), fetch(3)])
        collect_unordered([fetch(1), fetch(2)], concurrency=4)
    """
    it = iter(iterable)
    pending = _make_pending(it, concurrency)
    results: list[T] = []

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                result = task.result()
            except BaseException:
                for other in done:
                    if other is not task:
                        other.exception()  # mark as retrieved to avoid warnings
                await _cancel_all(pending, it)
                raise
            match result:
                case Ok(value):
                    results.append(value)
                    next_item = next(it, None)
                    if next_item is not None:
                        pending.add(asyncio.ensure_future(next_item))
                case Err() as err:
                    await _cancel_all(pending, it)
                    return err

    return Ok(results)


async def map_collect_unordered(
    iterable: Iterable[T],
    f: Callable[[T], _CoroOrTask[Result[U, E]]],
    *,
    concurrency: int | None = None,
) -> Result[list[U], E]:
    """
    Apply *f* to each element concurrently and collect into ``Ok[list]``.

    Results are returned in completion order, not input order.
    Returns the first ``Err`` produced by *f*, cancelling remaining tasks.

    *concurrency* limits how many calls to *f* run at the same time.
    ``None`` means unlimited — all are scheduled at once.

    **Exceptions**: if *f* raises, the exception propagates and all remaining
    tasks are cancelled. If multiple tasks raise in the same ``asyncio.wait`` batch,
    only one exception propagates — the rest are silently discarded.

    Example::

        map_collect_unordered(user_ids, fetch_user)
        map_collect_unordered(urls, fetch, concurrency=10)
    """
    return await collect_unordered(
        (f(element) for element in iterable),
        concurrency=concurrency,
    )


async def partition_unordered(
    iterable: Iterable[_CoroOrTask[Result[T, E]]],
    *,
    concurrency: int | None = None,
) -> tuple[list[T], list[E]]:
    """
    Await an iterable of coroutines or tasks concurrently, splitting results into ``(oks, errs)``.

    Results are collected in completion order, not input order.
    Unlike ``collect_unordered``, never short-circuits — all awaitables run to completion.

    *concurrency* limits how many run at the same time.
    ``None`` means unlimited — all are scheduled at once.

    **Exceptions**: if a coroutine raises, the exception propagates and all remaining
    tasks are cancelled. If multiple tasks raise in the same ``asyncio.wait`` batch,
    only one exception propagates — the rest are silently discarded.

    Example::

        oks, errs = await partition_unordered([fetch(1), fetch(2), fetch(3)])
        oks, errs = await partition_unordered([fetch(1), fetch(2)], concurrency=4)
    """
    it = iter(iterable)
    pending = _make_pending(it, concurrency)
    oks: list[T] = []
    errs: list[E] = []

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                result = task.result()
            except BaseException:
                for other in done:
                    if other is not task:
                        other.exception()  # mark as retrieved to avoid warnings
                await _cancel_all(pending, it)
                raise
            match result:
                case Ok(value):
                    oks.append(value)
                case Err(e):
                    errs.append(e)
            next_item = next(it, None)
            if next_item is not None:
                pending.add(asyncio.ensure_future(next_item))

    return oks, errs


async def filter_ok_unordered(
    iterable: Iterable[_CoroOrTask[Result[T, E]]],
    *,
    concurrency: int | None = None,
) -> AsyncIterator[T]:
    """
    Await coroutines or tasks concurrently, yielding ``Ok`` values as they complete.

    ``Err`` values are silently skipped.
    Values are yielded in completion order, not input order.

    *concurrency* limits how many run at the same time.
    ``None`` means unlimited — all are scheduled at once.

    **Exceptions**: if a coroutine raises, the exception propagates and all remaining
    tasks are cancelled. If multiple tasks raise in the same ``asyncio.wait`` batch,
    only one exception propagates — the rest are silently discarded.

    Example::

        async for user in filter_ok_unordered([fetch(1), fetch(2), fetch(3)]):
            print(user)
    """
    it = iter(iterable)
    pending = _make_pending(it, concurrency)

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                result = task.result()
            except BaseException:
                for other in done:
                    if other is not task:
                        other.exception()  # mark as retrieved to avoid warnings
                await _cancel_all(pending, it)
                raise
            match result:
                case Ok(value):
                    yield value
                case Err():
                    pass
            next_item = next(it, None)
            if next_item is not None:
                pending.add(asyncio.ensure_future(next_item))


async def filter_err_unordered(
    iterable: Iterable[_CoroOrTask[Result[T, E]]],
    *,
    concurrency: int | None = None,
) -> AsyncIterator[E]:
    """
    Await coroutines or tasks concurrently, yielding ``Err`` values as they complete.

    ``Ok`` values are silently skipped.
    Values are yielded in completion order, not input order.

    *concurrency* limits how many run at the same time.
    ``None`` means unlimited — all are scheduled at once.

    **Exceptions**: if a coroutine raises, the exception propagates and all remaining
    tasks are cancelled. If multiple tasks raise in the same ``asyncio.wait`` batch,
    only one exception propagates — the rest are silently discarded.

    Example::

        async for err in filter_err_unordered([fetch(1), fetch(2), fetch(3)]):
            print(err)
    """
    it = iter(iterable)
    pending = _make_pending(it, concurrency)

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                result = task.result()
            except BaseException:
                for other in done:
                    if other is not task:
                        other.exception()  # mark as retrieved to avoid warnings
                await _cancel_all(pending, it)
                raise
            match result:
                case Ok():
                    pass
                case Err(e):
                    yield e
            next_item = next(it, None)
            if next_item is not None:
                pending.add(asyncio.ensure_future(next_item))
