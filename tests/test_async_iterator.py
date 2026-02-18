from __future__ import annotations

import asyncio

import pytest

from corrode import Err, Ok
from corrode.async_iterator import (
    collect_unordered,
    filter_err_unordered,
    filter_ok_unordered,
    map_collect_unordered,
    partition_unordered,
)


async def ok_after(value: int, delay: float = 0.0) -> Ok[int]:
    await asyncio.sleep(delay)
    return Ok(value)


async def err_after(value: str, delay: float = 0.0) -> Err[str]:
    await asyncio.sleep(delay)
    return Err(value)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


class TestCollectUnorderedBasic:
    async def test_empty(self) -> None:
        result = await collect_unordered([])
        assert result == Ok([])

    async def test_all_ok(self) -> None:
        result = await collect_unordered([ok_after(1), ok_after(2), ok_after(3)])
        assert isinstance(result, Ok)
        assert sorted(result.ok_value) == [1, 2, 3]

    async def test_single_ok(self) -> None:
        assert await collect_unordered([ok_after(42)]) == Ok([42])

    async def test_single_err(self) -> None:
        assert await collect_unordered([err_after("boom")]) == Err("boom")

    async def test_first_err_returned(self) -> None:
        # both fail, first to complete wins
        result = await collect_unordered([
            err_after("first", delay=0.0),
            err_after("second", delay=0.1),
        ])
        assert result == Err("first")

    async def test_err_among_oks(self) -> None:
        result = await collect_unordered([ok_after(1), err_after("bad"), ok_after(3)])
        assert isinstance(result, Err)
        assert result.err_value == "bad"


# ---------------------------------------------------------------------------
# concurrency=None (unlimited)
# ---------------------------------------------------------------------------


class TestConcurrencyUnlimited:
    async def test_all_scheduled_at_once(self) -> None:
        # all tasks have the same delay — if they run concurrently total time ~delay,
        # not N*delay
        started_at: list[float] = []

        async def tracked(v: int) -> Ok[int]:
            started_at.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.05)
            return Ok(v)

        result = await collect_unordered(
            [tracked(i) for i in range(5)],
            concurrency=None,
        )
        assert isinstance(result, Ok)
        assert sorted(result.ok_value) == list(range(5))
        # all tasks started within a small window (< 0.02s)
        assert max(started_at) - min(started_at) < 0.02

    async def test_empty(self) -> None:
        assert await collect_unordered([], concurrency=None) == Ok([])

    async def test_err_cancels_pending(self) -> None:
        cancelled: list[int] = []

        async def slow_ok(v: int) -> Ok[int]:
            try:
                await asyncio.sleep(10)
                return Ok(v)
            except asyncio.CancelledError:
                cancelled.append(v)
                raise

        result = await collect_unordered(
            [err_after("boom", delay=0.0), slow_ok(1), slow_ok(2)],
            concurrency=None,
        )
        assert result == Err("boom")
        assert sorted(cancelled) == [1, 2]


# ---------------------------------------------------------------------------
# concurrency=1 (sequential)
# ---------------------------------------------------------------------------


class TestConcurrencyOne:
    async def test_all_ok(self) -> None:
        result = await collect_unordered(
            [ok_after(1), ok_after(2), ok_after(3)],
            concurrency=1,
        )
        assert isinstance(result, Ok)
        assert sorted(result.ok_value) == [1, 2, 3]

    async def test_runs_sequentially(self) -> None:
        order: list[int] = []

        async def tracked(v: int) -> Ok[int]:
            order.append(v)
            await asyncio.sleep(0)
            return Ok(v)

        await collect_unordered(
            [tracked(1), tracked(2), tracked(3)],
            concurrency=1,
        )
        # with concurrency=1 tasks are consumed one by one in input order
        assert order == [1, 2, 3]

    async def test_err_short_circuits(self) -> None:
        started: list[int] = []

        async def tracked_ok(v: int) -> Ok[int]:
            started.append(v)
            await asyncio.sleep(0)
            return Ok(v)

        result = await collect_unordered(
            [err_after("stop"), tracked_ok(1), tracked_ok(2)],
            concurrency=1,
        )
        assert result == Err("stop")
        # subsequent tasks were never started
        assert started == []

    async def test_empty(self) -> None:
        assert await collect_unordered([], concurrency=1) == Ok([])


# ---------------------------------------------------------------------------
# concurrency=N (sliding window)
# ---------------------------------------------------------------------------


class TestConcurrencyN:
    async def test_all_ok(self) -> None:
        result = await collect_unordered(
            [ok_after(i) for i in range(10)],
            concurrency=3,
        )
        assert isinstance(result, Ok)
        assert sorted(result.ok_value) == list(range(10))

    async def test_at_most_n_concurrent(self) -> None:
        in_flight: list[int] = []
        peak: list[int] = []

        async def tracked(v: int) -> Ok[int]:
            in_flight.append(v)
            peak.append(len(in_flight))
            await asyncio.sleep(0.02)
            in_flight.remove(v)
            return Ok(v)

        await collect_unordered(
            [tracked(i) for i in range(8)],
            concurrency=3,
        )
        assert max(peak) <= 3

    async def test_err_cancels_pending_window(self) -> None:
        cancelled: list[int] = []

        async def slow_ok(v: int) -> Ok[int]:
            try:
                await asyncio.sleep(10)
                return Ok(v)
            except asyncio.CancelledError:
                cancelled.append(v)
                raise

        result = await collect_unordered(
            [slow_ok(0), slow_ok(1), err_after("boom", delay=0.0), slow_ok(3)],
            concurrency=3,
        )
        assert result == Err("boom")
        # slow_ok(3) was never started (outside initial window of 3)
        assert 3 not in cancelled
        # slow_ok(0) and slow_ok(1) were cancelled
        assert sorted(cancelled) == [0, 1]

    async def test_concurrency_larger_than_input(self) -> None:
        result = await collect_unordered(
            [ok_after(i) for i in range(3)],
            concurrency=100,
        )
        assert isinstance(result, Ok)
        assert sorted(result.ok_value) == [0, 1, 2]

    async def test_empty(self) -> None:
        assert await collect_unordered([], concurrency=4) == Ok([])

    async def test_remaining_coros_not_started_after_err(self) -> None:
        # 6 coroutines, concurrency=3: first window is [0, 1, 2],
        # err fires immediately, coroutines [3, 4, 5] must never start
        started: list[int] = []

        async def slow(v: int) -> Ok[int]:
            started.append(v)
            await asyncio.sleep(10)
            return Ok(v)

        result = await collect_unordered(
            [err_after("stop"), slow(1), slow(2), slow(3), slow(4), slow(5)],
            concurrency=3,
        )
        assert result == Err("stop")
        assert all(v <= 2 for v in started)


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------


class TestExceptionPropagation:
    async def test_exception_escapes(self) -> None:
        async def boom() -> Ok[int]:
            raise ValueError("oops")

        with pytest.raises(ValueError, match="oops"):
            await collect_unordered([boom()])

    async def test_exception_cancels_pending(self) -> None:
        cancelled: list[int] = []

        async def slow(v: int) -> Ok[int]:
            try:
                await asyncio.sleep(10)
                return Ok(v)
            except asyncio.CancelledError:
                cancelled.append(v)
                raise

        async def boom() -> Ok[int]:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await collect_unordered([slow(1), slow(2), boom()])

        assert sorted(cancelled) == [1, 2]

    async def test_exception_closes_unconsumed_coros(self) -> None:
        started: list[int] = []

        async def boom() -> Ok[int]:
            raise RuntimeError

        async def never(v: int) -> Ok[int]:
            started.append(v)
            await asyncio.sleep(10)
            return Ok(v)

        with pytest.raises(RuntimeError):
            await collect_unordered([boom(), never(1), never(2)], concurrency=1)

        # with concurrency=1, never(1) and never(2) were never scheduled
        assert started == []

    async def test_exception_not_swallowed_when_err_also_present(self) -> None:
        # if one task raises and another returns Err, the exception wins
        async def boom() -> Ok[int]:
            raise RuntimeError("exception")

        with pytest.raises(RuntimeError, match="exception"):
            await collect_unordered([boom(), boom()], concurrency=None)


# ---------------------------------------------------------------------------
# create_task inputs
# ---------------------------------------------------------------------------


class TestCreateTaskInputs:
    async def test_all_ok_with_tasks(self) -> None:
        tasks = [asyncio.create_task(ok_after(i)) for i in range(4)]
        result = await collect_unordered(tasks)
        assert isinstance(result, Ok)
        assert sorted(result.ok_value) == [0, 1, 2, 3]

    async def test_err_cancels_out_of_window_tasks(self) -> None:
        # tasks are already running before collect_unordered is called
        cancelled = []

        async def slow(v: int) -> Ok[int]:
            try:
                await asyncio.sleep(10)
                return Ok(v)
            except asyncio.CancelledError:
                cancelled.append(v)
                raise

        tasks = [asyncio.create_task(slow(i)) for i in range(4)]
        await asyncio.sleep(0)  # let tasks start

        result = await collect_unordered(
            [err_after("stop"), *tasks],
            concurrency=2,
        )
        await asyncio.sleep(0)  # let cancellations propagate
        assert result == Err("stop")
        # all tasks are already-running Tasks (not plain coroutines),
        # so _cancel_all must call .cancel() on them, not .close()
        assert sorted(cancelled) == [0, 1, 2, 3]
        assert all(t.done() for t in tasks)

    async def test_mixed_coros_and_tasks(self) -> None:
        task = asyncio.create_task(ok_after(1))
        result = await collect_unordered([ok_after(0), task, ok_after(2)])
        assert isinstance(result, Ok)
        assert sorted(result.ok_value) == [0, 1, 2]


# ---------------------------------------------------------------------------
# partition_unordered
# ---------------------------------------------------------------------------


class TestPartitionUnordered:
    async def test_empty(self) -> None:
        assert await partition_unordered([]) == ([], [])

    async def test_all_ok(self) -> None:
        oks, errs = await partition_unordered([ok_after(i) for i in range(3)])
        assert sorted(oks) == [0, 1, 2]
        assert errs == []

    async def test_all_err(self) -> None:
        oks, errs = await partition_unordered([err_after(str(i)) for i in range(3)])
        assert oks == []
        assert sorted(errs) == ["0", "1", "2"]

    async def test_mixed(self) -> None:
        oks, errs = await partition_unordered([
            ok_after(1), err_after("a"), ok_after(2), err_after("b"),
        ])
        assert sorted(oks) == [1, 2]
        assert sorted(errs) == ["a", "b"]

    async def test_consumes_all_no_short_circuit(self) -> None:
        completed = []

        async def tracked(v: int) -> Ok[int]:
            await asyncio.sleep(0)
            completed.append(v)
            return Ok(v)

        await partition_unordered(
            [err_after("e"), tracked(1), tracked(2), tracked(3)],
        )
        assert sorted(completed) == [1, 2, 3]

    async def test_concurrency_none(self) -> None:
        started_at: list[float] = []

        async def tracked(v: int) -> Ok[int]:
            started_at.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.05)
            return Ok(v)

        oks, errs = await partition_unordered(
            [tracked(i) for i in range(5)],
            concurrency=None,
        )
        assert sorted(oks) == list(range(5))
        assert errs == []
        assert max(started_at) - min(started_at) < 0.02

    async def test_concurrency_one(self) -> None:
        order: list[int] = []

        async def tracked(v: int) -> Ok[int]:
            order.append(v)
            await asyncio.sleep(0)
            return Ok(v)

        await partition_unordered(
            [tracked(i) for i in range(4)],
            concurrency=1,
        )
        assert order == [0, 1, 2, 3]

    async def test_concurrency_n_at_most(self) -> None:
        in_flight: list[int] = []
        peak: list[int] = []

        async def tracked(v: int) -> Ok[int]:
            in_flight.append(v)
            peak.append(len(in_flight))
            await asyncio.sleep(0.02)
            in_flight.remove(v)
            return Ok(v)

        await partition_unordered(
            [tracked(i) for i in range(8)],
            concurrency=3,
        )
        assert max(peak) <= 3

    async def test_with_tasks(self) -> None:
        tasks = [asyncio.create_task(ok_after(i)) for i in range(3)]
        oks, errs = await partition_unordered(tasks)
        assert sorted(oks) == [0, 1, 2]
        assert errs == []

    async def test_exception_propagates(self) -> None:
        async def boom() -> Ok[int]:
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError, match="oops"):
            await partition_unordered([boom(), ok_after(1)])

    async def test_exception_cancels_pending(self) -> None:
        cancelled: list[int] = []

        async def slow(v: int) -> Ok[int]:
            try:
                await asyncio.sleep(10)
                return Ok(v)
            except asyncio.CancelledError:
                cancelled.append(v)
                raise

        async def boom() -> Ok[int]:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await partition_unordered([slow(1), slow(2), boom()])

        assert sorted(cancelled) == [1, 2]

    async def test_exception_drains_multiple_done(self) -> None:
        # two tasks finish simultaneously — if one raises, the other's
        # exception must be marked as retrieved to avoid warnings
        async def boom() -> Ok[int]:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await partition_unordered([boom(), boom()], concurrency=None)

    async def test_exception_closes_unconsumed_coros(self) -> None:
        started: list[int] = []

        async def boom() -> Ok[int]:
            raise RuntimeError

        async def never(v: int) -> Ok[int]:
            started.append(v)
            await asyncio.sleep(10)
            return Ok(v)

        with pytest.raises(RuntimeError):
            await partition_unordered([boom(), never(1), never(2)], concurrency=1)

        assert started == []


# ---------------------------------------------------------------------------
# map_collect_unordered
# ---------------------------------------------------------------------------


class TestMapCollectUnordered:
    async def test_all_ok(self) -> None:
        async def double(x: int) -> Ok[int]:
            await asyncio.sleep(0)
            return Ok(x * 2)

        result = await map_collect_unordered(range(4), double)
        assert isinstance(result, Ok)
        assert sorted(result.ok_value) == [0, 2, 4, 6]

    async def test_empty(self) -> None:
        async def double(x: int) -> Ok[int]:
            return Ok(x * 2)

        assert await map_collect_unordered([], double) == Ok([])

    async def test_first_err_short_circuits(self) -> None:
        called: list[int] = []

        async def maybe_fail(x: int) -> Ok[int] | Err[str]:
            called.append(x)
            await asyncio.sleep(0.01 * x)
            if x == 0:
                return Err("zero")
            return Ok(x)

        result = await map_collect_unordered(range(4), maybe_fail, concurrency=1)
        assert result == Err("zero")
        assert called == [0]

    async def test_concurrency_respected(self) -> None:
        in_flight: list[int] = []
        peak: list[int] = []

        async def tracked(x: int) -> Ok[int]:
            in_flight.append(x)
            peak.append(len(in_flight))
            await asyncio.sleep(0.02)
            in_flight.remove(x)
            return Ok(x)

        await map_collect_unordered(range(8), tracked, concurrency=3)
        assert max(peak) <= 3

    async def test_exception_propagates(self) -> None:
        async def boom(_x: int) -> Ok[int]:
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError, match="oops"):
            await map_collect_unordered([1, 2, 3], boom)


# ---------------------------------------------------------------------------
# filter_ok_unordered
# ---------------------------------------------------------------------------


class TestFilterOkUnordered:
    async def test_yields_ok_values(self) -> None:
        results = [
            v async for v in filter_ok_unordered([ok_after(1), err_after("x"), ok_after(2)])
        ]
        assert sorted(results) == [1, 2]

    async def test_empty(self) -> None:
        assert [v async for v in filter_ok_unordered([])] == []

    async def test_all_err_yields_nothing(self) -> None:
        assert [v async for v in filter_ok_unordered([err_after("a"), err_after("b")])] == []

    async def test_yields_as_completed(self) -> None:
        # fast completes before slow even though slow comes first in input
        order: list[int] = []
        async for v in filter_ok_unordered([ok_after(1, delay=0.1), ok_after(2, delay=0.0)]):
            order.append(v)
        assert order == [2, 1]

    async def test_concurrency_respected(self) -> None:
        in_flight: list[int] = []
        peak: list[int] = []

        async def tracked(v: int) -> Ok[int]:
            in_flight.append(v)
            peak.append(len(in_flight))
            await asyncio.sleep(0.02)
            in_flight.remove(v)
            return Ok(v)

        async for _ in filter_ok_unordered([tracked(i) for i in range(8)], concurrency=3):
            pass
        assert max(peak) <= 3

    async def test_exception_propagates(self) -> None:
        async def boom() -> Ok[int]:
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError, match="oops"):
            async for _ in filter_ok_unordered([boom(), ok_after(1)]):
                pass


# ---------------------------------------------------------------------------
# filter_err_unordered
# ---------------------------------------------------------------------------


class TestFilterErrUnordered:
    async def test_yields_err_values(self) -> None:
        results = [
            e async for e in filter_err_unordered([ok_after(1), err_after("x"), err_after("y")])
        ]
        assert sorted(results) == ["x", "y"]

    async def test_empty(self) -> None:
        assert [e async for e in filter_err_unordered([])] == []

    async def test_all_ok_yields_nothing(self) -> None:
        assert [e async for e in filter_err_unordered([ok_after(1), ok_after(2)])] == []

    async def test_yields_as_completed(self) -> None:
        order: list[str] = []
        async for e in filter_err_unordered(
            [err_after("slow", delay=0.1), err_after("fast", delay=0.0)],
        ):
            order.append(e)
        assert order == ["fast", "slow"]

    async def test_concurrency_respected(self) -> None:
        in_flight: list[int] = []
        peak: list[int] = []

        async def tracked(v: int) -> Err[int]:
            in_flight.append(v)
            peak.append(len(in_flight))
            await asyncio.sleep(0.02)
            in_flight.remove(v)
            return Err(v)

        async for _ in filter_err_unordered([tracked(i) for i in range(8)], concurrency=3):
            pass
        assert max(peak) <= 3

    async def test_exception_propagates(self) -> None:
        async def boom() -> Err[str]:
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError, match="oops"):
            async for _ in filter_err_unordered([boom(), err_after("x")]):
                pass
