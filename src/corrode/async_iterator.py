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
_S = TypeVar("_S")
_CoroOrTask = Coroutine[object, object, _R] | asyncio.Task[_R]


async def _cancel_all(
    pending: set[asyncio.Task[_R]],
    it: Iterator[_CoroOrTask[_S]],
) -> None:
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for item in it:
        if isinstance(item, asyncio.Task):
            item.cancel()
        else:
            item.close()


def _wrap_indexed(idx: int, item: _CoroOrTask[_R]) -> asyncio.Task[tuple[int, _R]]:
    async def _inner() -> tuple[int, _R]:
        return idx, await item

    return asyncio.ensure_future(_inner())


def _make_pending_indexed(
    it: Iterator[_CoroOrTask[_R]],
    concurrency: int | None,
) -> tuple[set[asyncio.Task[tuple[int, _R]]], int]:
    pending: set[asyncio.Task[tuple[int, _R]]] = set()
    idx = 0
    for item in (it if concurrency is None else itertools.islice(it, concurrency)):
        pending.add(_wrap_indexed(idx, item))
        idx += 1
    return pending, idx


async def collect(
    iterable: Iterable[_CoroOrTask[Result[T, E]]],
    *,
    concurrency: int | None = None,
) -> Result[list[T], E]:
    """
    Await an iterable of coroutines or tasks concurrently, collecting results into ``Ok[list]``.

    Results are returned in input order.
    Returns the first ``Err`` encountered, cancelling remaining tasks.

    *concurrency* limits how many run at the same time.
    ``None`` means unlimited — all are scheduled at once.

    **Exceptions**: if a coroutine raises, the exception propagates and all remaining
    tasks are cancelled. If multiple tasks raise in the same ``asyncio.wait`` batch,
    only one exception propagates — the rest are silently discarded.

    Example::

        collect([fetch(1), fetch(2), fetch(3)])
        collect([fetch(1), fetch(2)], concurrency=4)
    """
    it = iter(iterable)
    pending, next_idx = _make_pending_indexed(it, concurrency)
    indexed: dict[int, T] = {}

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                idx, result = task.result()
            except BaseException:
                for other in done:
                    if other is not task:
                        other.exception()  # mark as retrieved to avoid warnings
                await _cancel_all(pending, it)
                raise
            match result:
                case Ok(value):
                    indexed[idx] = value
                    next_item = next(it, None)
                    if next_item is not None:
                        pending.add(_wrap_indexed(next_idx, next_item))
                        next_idx += 1
                case Err() as err:
                    await _cancel_all(pending, it)
                    return err

    return Ok([indexed[i] for i in range(len(indexed))])


async def map_collect(
    iterable: Iterable[T],
    f: Callable[[T], _CoroOrTask[Result[U, E]]],
    *,
    concurrency: int | None = None,
) -> Result[list[U], E]:
    """
    Apply *f* to each element concurrently and collect into ``Ok[list]``.

    Results are returned in input order.
    Returns the first ``Err`` produced by *f*, cancelling remaining tasks.

    *concurrency* limits how many calls to *f* run at the same time.
    ``None`` means unlimited — all are scheduled at once.

    **Exceptions**: if *f* raises, the exception propagates and all remaining
    tasks are cancelled. If multiple tasks raise in the same ``asyncio.wait`` batch,
    only one exception propagates — the rest are silently discarded.

    Example::

        map_collect(user_ids, fetch_user)
        map_collect(urls, fetch, concurrency=10)
    """
    return await collect(
        (f(element) for element in iterable),
        concurrency=concurrency,
    )


async def partition(
    iterable: Iterable[_CoroOrTask[Result[T, E]]],
    *,
    concurrency: int | None = None,
) -> tuple[list[T], list[E]]:
    """
    Await an iterable of coroutines or tasks concurrently, splitting results into ``(oks, errs)``.

    Results are collected in input order within each list.
    Unlike ``collect``, never short-circuits — all awaitables run to completion.

    *concurrency* limits how many run at the same time.
    ``None`` means unlimited — all are scheduled at once.

    **Exceptions**: if a coroutine raises, the exception propagates and all remaining
    tasks are cancelled. If multiple tasks raise in the same ``asyncio.wait`` batch,
    only one exception propagates — the rest are silently discarded.

    Example::

        oks, errs = await partition([fetch(1), fetch(2), fetch(3)])
        oks, errs = await partition([fetch(1), fetch(2)], concurrency=4)
    """
    it = iter(iterable)
    pending, next_idx = _make_pending_indexed(it, concurrency)
    indexed: dict[int, Result[T, E]] = {}

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                idx, result = task.result()
            except BaseException:
                for other in done:
                    if other is not task:
                        other.exception()  # mark as retrieved to avoid warnings
                await _cancel_all(pending, it)
                raise
            indexed[idx] = result
            next_item = next(it, None)
            if next_item is not None:
                pending.add(_wrap_indexed(next_idx, next_item))
                next_idx += 1

    oks: list[T] = []
    errs: list[E] = []
    for result in (indexed[i] for i in range(len(indexed))):
        match result:
            case Ok(value):
                oks.append(value)
            case Err(e):
                errs.append(e)
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
    pending: set[asyncio.Task[Result[T, E]]] = {
        asyncio.ensure_future(item)
        for item in (it if concurrency is None else itertools.islice(it, concurrency))
    }

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
    pending: set[asyncio.Task[Result[T, E]]] = {
        asyncio.ensure_future(item)
        for item in (it if concurrency is None else itertools.islice(it, concurrency))
    }

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


async def filter_ok(
    iterable: Iterable[_CoroOrTask[Result[T, E]]],
    *,
    concurrency: int,
) -> AsyncIterator[T]:
    """
    Await coroutines or tasks concurrently, yielding ``Ok`` values in input order.

    ``Err`` values are silently skipped.
    Values are yielded in input order — later-completing tasks are buffered until
    all earlier ones have been yielded.

    *concurrency* controls the size of the sliding window of in-flight tasks.
    Unlike ``filter_ok_unordered``, ``None`` is not accepted because the
    reorder buffer would be unbounded.

    **Exceptions**: if a coroutine raises, the exception propagates and all remaining
    tasks are cancelled. If multiple tasks raise in the same ``asyncio.wait`` batch,
    only one exception propagates — the rest are silently discarded.

    Example::

        async for user in filter_ok([fetch(1), fetch(2), fetch(3)], concurrency=4):
            print(user)
    """
    it = iter(iterable)
    pending, next_idx = _make_pending_indexed(it, concurrency)
    buf: dict[int, Result[T, E]] = {}
    next_yield = 0

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                idx, result = task.result()
            except BaseException:
                for other in done:
                    if other is not task:
                        other.exception()
                await _cancel_all(pending, it)
                raise
            buf[idx] = result
            next_item = next(it, None)
            if next_item is not None:
                pending.add(_wrap_indexed(next_idx, next_item))
                next_idx += 1

        while next_yield in buf:
            match buf.pop(next_yield):
                case Ok(value):
                    yield value
                case Err():
                    pass
            next_yield += 1


async def filter_err(
    iterable: Iterable[_CoroOrTask[Result[T, E]]],
    *,
    concurrency: int,
) -> AsyncIterator[E]:
    """
    Await coroutines or tasks concurrently, yielding ``Err`` values in input order.

    ``Ok`` values are silently skipped.
    Values are yielded in input order — later-completing tasks are buffered until
    all earlier ones have been yielded.

    *concurrency* controls the size of the sliding window of in-flight tasks.
    Unlike ``filter_err_unordered``, ``None`` is not accepted because the
    reorder buffer would be unbounded.

    **Exceptions**: if a coroutine raises, the exception propagates and all remaining
    tasks are cancelled. If multiple tasks raise in the same ``asyncio.wait`` batch,
    only one exception propagates — the rest are silently discarded.

    Example::

        async for err in filter_err([fetch(1), fetch(2), fetch(3)], concurrency=4):
            print(err)
    """
    it = iter(iterable)
    pending, next_idx = _make_pending_indexed(it, concurrency)
    buf: dict[int, Result[T, E]] = {}
    next_yield = 0

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                idx, result = task.result()
            except BaseException:
                for other in done:
                    if other is not task:
                        other.exception()
                await _cancel_all(pending, it)
                raise
            buf[idx] = result
            next_item = next(it, None)
            if next_item is not None:
                pending.add(_wrap_indexed(next_idx, next_item))
                next_idx += 1

        while next_yield in buf:
            match buf.pop(next_yield):
                case Ok():
                    pass
                case Err(e):
                    yield e
            next_yield += 1


async def try_reduce(
    iterable: Iterable[_CoroOrTask[T]],
    initial: U,
    f: Callable[[U, T], Result[U, E]],
) -> Result[U, E]:
    """
    Await each coroutine or task sequentially, folding with *f* and short-circuiting on ``Err``.

    Unlike the async ``collect`` / ``partition`` family, tasks run one at a time —
    each awaited value is passed to *f* before the next is awaited, because the
    accumulator depends on the previous step.

    Example::

        async def fetch_int(url: str) -> int: ...

        def safe_add(acc: int, x: int) -> Result[int, str]:
            if x < 0:
                return Err(f"negative: {x}")
            return Ok(acc + x)

        await try_reduce([fetch_int(u1), fetch_int(u2)], 0, safe_add)
    """
    it = iter(iterable)
    acc: U = initial
    for item in it:
        value: T = await item
        match f(acc, value):
            case Ok(new_acc):
                acc = new_acc
            case Err() as err:
                for remaining in it:
                    if isinstance(remaining, asyncio.Task):
                        remaining.cancel()
                    else:
                        remaining.close()
                return err
    return Ok(acc)
