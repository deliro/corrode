from __future__ import annotations

from corrode import (
    Err,
    Ok,
    Result,
    collect,
    filter_err,
    filter_ok,
    map_collect,
    partition,
    try_reduce,
)

# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


class TestCollect:
    def test_all_ok(self) -> None:
        assert collect([Ok(1), Ok(2), Ok(3)]) == Ok([1, 2, 3])

    def test_empty(self) -> None:
        assert collect([]) == Ok([])

    def test_first_err_short_circuits(self) -> None:
        calls = 0

        def counted_iter():
            nonlocal calls
            for r in [Ok(1), Err("bad"), Ok(3)]:
                calls += 1
                yield r

        result = collect(counted_iter())
        assert result == Err("bad")
        assert calls == 2  # stopped after the Err

    def test_first_err_returned(self) -> None:
        assert collect([Ok(1), Err("first"), Err("second")]) == Err("first")

    def test_only_err(self) -> None:
        assert collect([Err(42)]) == Err(42)

    def test_generator_input(self) -> None:
        assert collect(Ok(x) for x in range(3)) == Ok([0, 1, 2])


# ---------------------------------------------------------------------------
# map_collect
# ---------------------------------------------------------------------------


def _parse(s: str) -> Result[int, str]:
    try:
        return Ok(int(s))
    except ValueError:
        return Err(f"not a number: {s!r}")


class TestMapCollect:
    def test_all_ok(self) -> None:
        assert map_collect(["1", "2", "3"], _parse) == Ok([1, 2, 3])

    def test_empty(self) -> None:
        assert map_collect([], _parse) == Ok([])

    def test_first_err_short_circuits(self) -> None:
        calls = 0

        def counting_parse(s: str) -> Result[int, str]:
            nonlocal calls
            calls += 1
            return _parse(s)

        result = map_collect(["1", "x", "3"], counting_parse)
        assert result == Err("not a number: 'x'")
        assert calls == 2

    def test_first_err_returned(self) -> None:
        result = map_collect(["x", "y"], _parse)
        assert result == Err("not a number: 'x'")

    def test_generator_input(self) -> None:
        assert map_collect((str(x) for x in range(3)), _parse) == Ok([0, 1, 2])


# ---------------------------------------------------------------------------
# partition
# ---------------------------------------------------------------------------


class TestPartition:
    def test_mixed(self) -> None:
        oks, errs = partition([Ok(1), Err("a"), Ok(2), Err("b")])
        assert oks == [1, 2]
        assert errs == ["a", "b"]

    def test_all_ok(self) -> None:
        oks, errs = partition([Ok(1), Ok(2)])
        assert oks == [1, 2]
        assert errs == []

    def test_all_err(self) -> None:
        oks, errs = partition([Err("x"), Err("y")])
        assert oks == []
        assert errs == ["x", "y"]

    def test_empty(self) -> None:
        oks, errs = partition([])
        assert oks == []
        assert errs == []

    def test_consumes_all_elements(self) -> None:
        consumed = []

        def tracked_iter():
            for r in [Ok(1), Err("a"), Ok(2)]:
                consumed.append(r)
                yield r

        partition(tracked_iter())
        assert len(consumed) == 3

    def test_preserves_order(self) -> None:
        oks, errs = partition([Ok(3), Ok(1), Err("b"), Err("a")])
        assert oks == [3, 1]
        assert errs == ["b", "a"]


# ---------------------------------------------------------------------------
# filter_ok
# ---------------------------------------------------------------------------


class TestFilterOk:
    def test_mixed(self) -> None:
        assert list(filter_ok([Ok(1), Err("x"), Ok(2)])) == [1, 2]

    def test_all_err(self) -> None:
        assert list(filter_ok([Err("a"), Err("b")])) == []

    def test_all_ok(self) -> None:
        assert list(filter_ok([Ok(1), Ok(2), Ok(3)])) == [1, 2, 3]

    def test_empty(self) -> None:
        assert list(filter_ok([])) == []

    def test_lazy(self) -> None:
        # filter_ok should be a generator — check it doesn't eagerly consume
        it = filter_ok(Ok(x) for x in range(3))
        assert next(it) == 0
        assert next(it) == 1


# ---------------------------------------------------------------------------
# filter_err
# ---------------------------------------------------------------------------


class TestFilterErr:
    def test_mixed(self) -> None:
        assert list(filter_err([Ok(1), Err("x"), Ok(2), Err("y")])) == ["x", "y"]

    def test_all_ok(self) -> None:
        assert list(filter_err([Ok(1), Ok(2)])) == []

    def test_all_err(self) -> None:
        assert list(filter_err([Err("a"), Err("b")])) == ["a", "b"]

    def test_empty(self) -> None:
        assert list(filter_err([])) == []

    def test_lazy(self) -> None:
        it = filter_err(Err(x) for x in range(3))
        assert next(it) == 0
        assert next(it) == 1


# ---------------------------------------------------------------------------
# try_reduce
# ---------------------------------------------------------------------------


def _safe_add(acc: int, x: int) -> Result[int, str]:
    if x < 0:
        return Err(f"negative: {x}")
    return Ok(acc + x)


class TestTryReduce:
    def test_all_ok(self) -> None:
        assert try_reduce([1, 2, 3], 0, _safe_add) == Ok(6)

    def test_empty(self) -> None:
        assert try_reduce([], 0, _safe_add) == Ok(0)

    def test_short_circuits_on_err(self) -> None:
        calls = 0

        def counting_add(acc: int, x: int) -> Result[int, str]:
            nonlocal calls
            calls += 1
            return _safe_add(acc, x)

        result = try_reduce([1, -1, 3], 0, counting_add)
        assert result == Err("negative: -1")
        assert calls == 2

    def test_first_err_returned(self) -> None:
        result = try_reduce([-1, -2], 0, _safe_add)
        assert result == Err("negative: -1")

    def test_initial_value_used(self) -> None:
        assert try_reduce([1, 2], 10, _safe_add) == Ok(13)

    def test_accumulator_updated_correctly(self) -> None:
        def concat(acc: str, x: str) -> Result[str, str]:
            if not x:
                return Err("empty string")
            return Ok(acc + x)

        assert try_reduce(["a", "b", "c"], "", concat) == Ok("abc")
