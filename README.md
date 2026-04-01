# corrode

A Rust-like `Result` type for Python 3.11+, fully type annotated.

<div align="center">

> *Explicit is better than implicit.*
> *Errors should never pass silently.*
>
> — The Zen of Python

[![CI](https://github.com/deliro/corrode/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/deliro/corrode/actions/workflows/ci.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/deliro/corrode/branch/main/graph/badge.svg)](https://codecov.io/gh/deliro/corrode)

</div>

## Table of Contents

- [Installation](#installation)
- [Why](#why)
- [Exhaustive error handling](#exhaustive-error-handling)
- [Adopting corrode in an existing codebase](#adopting-corrode-in-an-existing-codebase)
- [API reference](#api-reference)
  - [Pattern matching](#pattern-matching)
  - [Transforming values](#transforming-values)
  - [Chaining with `and_then` / `or_else`](#chaining-with-and_then--or_else)
  - [Combining results with `zip`](#combining-results-with-zip)
  - [Predicates](#predicates)
  - [Inspecting](#inspecting)
  - [Async methods](#async-methods)
  - [`do` notation (not recommended)](#do-notation-not-recommended)
  - [`@as_result` / `@as_async_result`](#as_result--as_async_result)
  - [Escape hatches](#escape-hatches)
- [Iterator utilities](#iterator-utilities)
  - [`collect`](#collect)
  - [`map_collect`](#map_collect)
  - [`partition`](#partition)
  - [`filter_ok`](#filter_ok)
  - [`filter_err`](#filter_err)
  - [`try_reduce`](#try_reduce)
- [Async iterator utilities](#async-iterator-utilities)
  - [`collect`](#collect)
  - [`map_collect`](#map_collect)
  - [`partition`](#partition)
  - [`filter_ok_unordered`](#filter_ok_unordered)
  - [`filter_err_unordered`](#filter_err_unordered)
  - [`filter_ok`](#filter_ok-1)
  - [`filter_err`](#filter_err-1)
  - [`try_reduce`](#try_reduce-1)
- [License](#license)

## Installation

```sh
uv add corrode
```

or with pip / poetry:

```sh
pip install corrode
poetry add corrode
```

## Why

Exceptions are implicit. Nothing in a function signature tells you it can
raise, what it raises, or whether the caller remembered to handle it.
Bugs hide until production, and `except Exception` becomes the norm:

```python
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str

# Can this raise? What exceptions? The signature doesn't tell you.
def get_user(user_id: int) -> User:
    if user_id <= 0:
        raise ValueError(f"Invalid user ID: {user_id}")
    if user_id == 13:
        raise PermissionError("Access denied")
    return User(id=user_id, name="Alice")

# The caller has no idea this can fail — until it does in production
user = get_user(1)
assert user.name == "Alice"
```

`Result[T, E]` is a union of `Ok[T] | Err[E]`:

```python
from dataclasses import dataclass
from corrode import Result, Ok, Err

@dataclass
class User:
    id: int
    name: str

@dataclass
class NotFound:
    user_id: int

@dataclass
class Forbidden:
    reason: str

type GetUserError = NotFound | Forbidden

# Errors are now part of the return type — callers see exactly what can go wrong
def get_user(user_id: int) -> Result[User, GetUserError]:
    # Instead of raise, return Err — the type checker tracks it
    if user_id <= 0:
        return Err(NotFound(user_id=user_id))
    if user_id == 13:
        return Err(Forbidden(reason="banned"))
    return Ok(User(id=user_id, name="Alice"))

# Can't ignore errors — Result forces you to handle both variants
assert get_user(1) == Ok(User(id=1, name="Alice"))
assert get_user(-1) == Err(NotFound(user_id=-1))
```

## Exhaustive error handling

```python
from dataclasses import dataclass
from typing import assert_never
from corrode import Ok, Err, Result


@dataclass
class User:
    id: int
    name: str

@dataclass
class NotFound:
    user_id: int

@dataclass
class Forbidden:
    reason: str

type GetUserError = NotFound | Forbidden


def get_user(user_id: int) -> Result[User, GetUserError]:
    if user_id <= 0:
        return Err(NotFound(user_id=user_id))
    if user_id == 13:
        return Err(Forbidden(reason="banned"))
    return Ok(User(id=user_id, name="Alice"))


match get_user(42):
    case Ok(user):
        print(f"Welcome, {user.name}")
    case Err(e):
        # Nested match on the error — each variant handled explicitly
        match e:
            case NotFound(user_id=uid):
                print(f"User {uid} does not exist")
            case Forbidden(reason=reason):
                print(f"Access denied: {reason}")
            case _:
                # If you add a new variant to GetUserError, mypy reports
                # an error here until you handle it — compile-time safety
                assert_never(e)
```

## Adopting corrode in an existing codebase

You don't have to rewrite everything at once. Exceptions don't disappear
overnight, and third-party libraries will always raise them. That's fine —
`corrode` is designed for gradual adoption.

### Step 1: wrap existing functions with `@as_result`

```python
import os
from corrode import as_result, Ok, Err

# Before: raises KeyError, ValueError, nobody knows about it
def parse_port_unsafe(key: str) -> int:
    return int(os.environ[key])

# After: just add the decorator — body unchanged, callers get Result
@as_result(KeyError, ValueError)
def parse_port(key: str) -> int:
    return int(os.environ[key])


def start_server(port: int) -> None:
    pass  # placeholder


os.environ["PORT"] = "3000"

match parse_port("PORT"):
    case Ok(port):
        start_server(port)
    case Err(KeyError()):
        start_server(8080)
    case Err(ValueError() as e):
        print(f"Invalid PORT: {e}")
```

### Step 2: return `Err(exception)` explicitly

```python
import os
from corrode import Ok, Err, Result


# @as_result removed — errors returned explicitly now.
# Return type visible in the signature, not hidden in the decorator.
# Callers don't change — they still match on the same exception types.
def parse_port(key: str) -> Result[int, KeyError | ValueError]:
    raw = os.environ.get(key)
    if raw is None:
        return Err(KeyError(key))
    try:
        return Ok(int(raw))
    except ValueError as exc:
        return Err(exc)


os.environ["PORT"] = "8080"
assert parse_port("PORT") == Ok(8080)
assert isinstance(parse_port("MISSING").err(), KeyError)
```

### Step 3: replace exceptions with domain types

```python
import os
from dataclasses import dataclass
from corrode import Ok, Err, Result


# Exception classes replaced with dataclasses —
# each error carries exactly the data the caller needs,
# no more parsing exception messages to figure out what went wrong.
@dataclass
class MissingKey:
    key: str

@dataclass
class InvalidValue:
    key: str
    raw: str

type ConfigError = MissingKey | InvalidValue


def parse_port(key: str) -> Result[int, ConfigError]:
    raw = os.environ.get(key)
    if raw is None:
        return Err(MissingKey(key=key))
    try:
        return Ok(int(raw))
    except ValueError:
        return Err(InvalidValue(key=key, raw=raw))


os.environ["PORT"] = "8080"
assert parse_port("PORT") == Ok(8080)
assert parse_port("MISSING") == Err(MissingKey(key="MISSING"))
```

### Exceptions inside Result-returning code

```python
import json
from dataclasses import dataclass
from corrode import Ok, Err, Result


@dataclass
class InvalidJson:
    raw: str

@dataclass
class MissingField:
    field: str


def parse_config(raw: str) -> Result[dict[str, int], InvalidJson | MissingField]:
    # try/except and Result mix freely in the same function —
    # just catch what you caught before and wrap it in Err
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return Err(InvalidJson(raw=raw))

    # From here on, plain Result logic
    if "port" not in data:
        return Err(MissingField(field="port"))
    return Ok(data)


assert parse_config('{"port": 8080}') == Ok({"port": 8080})
assert parse_config("not json") == Err(InvalidJson(raw="not json"))
assert parse_config("{}") == Err(MissingField(field="port"))
```

## API reference

### Pattern matching

```python
from dataclasses import dataclass
from typing import assert_never
from corrode import Ok, Err, Result


@dataclass
class User:
    id: int
    name: str
    balance: int


@dataclass
class NotFound:
    user_id: int


@dataclass
class InsufficientFunds:
    have: int
    need: int


type PaymentError = NotFound | InsufficientFunds


def get_user(user_id: int) -> Result[User, NotFound]:
    if user_id == 42:
        return Ok(User(id=42, name="Alice", balance=100))
    return Err(NotFound(user_id=user_id))


def charge(user: User, amount: int) -> Result[User, InsufficientFunds]:
    if user.balance < amount:
        return Err(InsufficientFunds(have=user.balance, need=amount))
    return Ok(User(id=user.id, name=user.name, balance=user.balance - amount))


def process_payment(user_id: int, amount: int) -> Result[User, PaymentError]:
    # Pattern matching to chain fallible operations
    match get_user(user_id):
        case Err(e):
            return Err(e)
        case Ok(user):
            return charge(user, amount)


match process_payment(42, 50):
    case Ok(user):
        print(f"{user.name} charged, new balance: {user.balance}")
    case Err(e):
        match e:
            case NotFound(user_id=uid):
                print(f"User {uid} not found")
            case InsufficientFunds(have=h, need=n):
                print(f"Need {n}, but only have {h}")
            case _:
                assert_never(e)
```

### Transforming values

```python
from dataclasses import dataclass
from corrode import Ok, Err, Result


@dataclass
class User:
    id: int
    name: str


@dataclass
class ApiError:
    code: int
    message: str


def get_user(user_id: int) -> Result[User, ApiError]:
    if user_id == 42:
        return Ok(User(id=42, name="Alice"))
    return Err(ApiError(code=404, message="User not found"))


def get_name(user: User) -> str:
    return user.name


def format_error(err: ApiError) -> str:
    return f"Error {err.code}: {err.message}"


# map transforms Ok, passes Err through unchanged
assert get_user(42).map(get_name) == Ok("Alice")
assert get_user(0).map(get_name) == Err(ApiError(code=404, message="User not found"))

# map_err transforms Err, passes Ok through unchanged
assert get_user(0).map_err(format_error) == Err("Error 404: User not found")
assert get_user(42).map_err(format_error) == Ok(User(id=42, name="Alice"))

# Get the value or a default
assert get_user(42).map_or("Unknown", get_name) == "Alice"
assert get_user(0).map_or("Unknown", get_name) == "Unknown"

# Compute default from the error
def error_placeholder(err: ApiError) -> str:
    return f"User #{err.code}"


assert get_user(0).map_or_else(error_placeholder, get_name) == "User #404"
```

Async variants: `map_async`, `map_err_async`, `map_or_async`, `map_or_else_async`.

### Chaining with `and_then` / `or_else`

```python
from dataclasses import dataclass
from corrode import Ok, Err, Result


@dataclass
class User:
    id: int
    name: str
    email: str


@dataclass
class ValidationError:
    field: str
    message: str


def parse_email(email: str) -> Result[str, ValidationError]:
    if "@" not in email:
        return Err(ValidationError(field="email", message="Invalid email format"))
    return Ok(email.lower().strip())


def parse_name(name: str) -> Result[str, ValidationError]:
    if len(name) < 2:
        return Err(ValidationError(field="name", message="Name too short"))
    return Ok(name.strip())


def create_user(user_id: int, name: str, email: str) -> Result[User, ValidationError]:
    # Sequential match — each step short-circuits on Err
    match parse_name(name):
        case Ok(n):
            pass
        case err:
            return err
    match parse_email(email):
        case Ok(e):
            return Ok(User(id=user_id, name=n, email=e))
        case err:
            return err


assert create_user(1, "Alice", "alice@example.com") == Ok(User(id=1, name="Alice", email="alice@example.com"))
assert create_user(1, "A", "alice@example.com") == Err(ValidationError(field="name", message="Name too short"))
assert create_user(1, "Alice", "invalid") == Err(ValidationError(field="email", message="Invalid email format"))
```

```python
from corrode import Ok, Err, Result


def fetch_from_cache(key: str) -> Result[str, str]:
    return Err("cache miss")


def fetch_from_db(key: str) -> Result[str, str]:
    if key == "user:1":
        return Ok("Alice")
    return Err("not found in db")


def fetch_from_api(key: str) -> Result[str, str]:
    return Ok("fetched from API")


# or_else provides recovery — try cache, then DB, then API
result = (
    fetch_from_cache("user:1")
    .or_else(lambda _: fetch_from_db("user:1"))
    .or_else(lambda _: fetch_from_api("user:1"))
)
assert result == Ok("Alice")  # Found in DB
```

Async variants: `and_then_async`, `or_else_async`.

### Combining results with `zip`

Combine two to five independent `Result` values into a single `Ok` tuple.
Returns the first `Err` encountered if any result fails:

```python
from corrode import Ok, Err, Result


def parse_int(s: str) -> Result[int, str]:
    return Ok(int(s)) if s.isdigit() else Err(f"not a number: {s!r}")


def parse_float(s: str) -> Result[float, str]:
    try:
        return Ok(float(s))
    except ValueError:
        return Err(f"not a float: {s!r}")


# All Ok — get a tuple
assert parse_int("3").zip(parse_float("1.5")) == Ok((3, 1.5))

# Any Err — get the first error
assert parse_int("x").zip(parse_float("1.5")) == Err("not a number: 'x'")
assert parse_int("3").zip(parse_float("y")) == Err("not a float: 'y'")

# Works with up to four extra arguments
assert Ok(1).zip(Ok(2), Ok(3), Ok(4)) == Ok((1, 2, 3, 4))
```

`Err.zip` always returns `self` without inspecting the other arguments:

```python
from corrode import Ok, Err

assert Err("already failed").zip(Ok(1), Ok(2)) == Err("already failed")
```

### Predicates

Check conditions on the contained value without unwrapping:

```python
from dataclasses import dataclass
from corrode import Ok, Err, Result


@dataclass
class User:
    id: int
    is_admin: bool


def get_user(user_id: int) -> Result[User, str]:
    if user_id == 1:
        return Ok(User(id=1, is_admin=True))
    if user_id == 2:
        return Ok(User(id=2, is_admin=False))
    return Err("not found")


def check_admin(user: User) -> bool:
    return user.is_admin


def is_not_found(err: str) -> bool:
    return "not found" in err


# Check if result is Ok AND satisfies a condition
assert get_user(1).is_ok_and(check_admin) is True
assert get_user(2).is_ok_and(check_admin) is False
assert get_user(99).is_ok_and(check_admin) is False

# Check if result is Err AND satisfies a condition
assert get_user(99).is_err_and(is_not_found) is True
assert get_user(1).is_err_and(is_not_found) is False
```

Async variants: `is_ok_and_async`, `is_err_and_async`.

### Inspecting

Perform side effects (logging, metrics) without consuming the result:

```python
from dataclasses import dataclass
from corrode import Ok, Err, Result


@dataclass
class User:
    id: int
    name: str


logs: list[str] = []


def log_success(user: User) -> None:
    logs.append(f"Found user: {user.name}")


def log_error(error: str) -> None:
    logs.append(f"Error: {error}")


def get_user(user_id: int) -> Result[User, str]:
    if user_id == 42:
        return Ok(User(id=42, name="Alice"))
    return Err("not found")


# Logs are written, but the result passes through unchanged
result = get_user(42).inspect(log_success).inspect_err(log_error)
assert result == Ok(User(id=42, name="Alice"))
assert logs == ["Found user: Alice"]

logs.clear()

result = get_user(0).inspect(log_success).inspect_err(log_error)
assert result == Err("not found")
assert logs == ["Error: not found"]
```

Async variants: `inspect_async`, `inspect_err_async`.

### Async methods

All transformation methods have `_async` variants for async callbacks:

```python
import asyncio
from dataclasses import dataclass
from corrode import Ok, Err, Result


@dataclass
class User:
    id: int
    name: str


@dataclass
class Profile:
    bio: str


async def fetch_profile(user: User) -> Profile:
    # Simulate async I/O
    return Profile(bio=f"Bio for {user.name}")


async def validate_user(user: User) -> Result[User, str]:
    if user.id <= 0:
        return Err("Invalid user ID")
    return Ok(user)


async def main() -> None:
    user_result: Result[User, str] = Ok(User(id=42, name="Alice"))

    # Async map
    profile_result = await user_result.map_async(fetch_profile)
    assert profile_result == Ok(Profile(bio="Bio for Alice"))

    # Async and_then
    validated = await user_result.and_then_async(validate_user)
    assert validated == Ok(User(id=42, name="Alice"))


asyncio.run(main())
```

Full list: `map_async`, `map_err_async`, `map_or_async`, `map_or_else_async`,
`and_then_async`, `or_else_async`, `is_ok_and_async`, `is_err_and_async`,
`inspect_async`, `inspect_err_async`.

### `do` notation (not recommended)

> **Not recommended.** Prefer `match`, `and_then()` chains, or `zip()` — all
> of which are fully typed without annotations.

Syntactic sugar for a sequence of `and_then()` calls. If any step is `Err`,
the whole expression short-circuits:

```python
from dataclasses import dataclass
from corrode import do, Ok, Err, Result


@dataclass
class User:
    id: int
    name: str


@dataclass
class NotFound:
    pass


def get_user(user_id: int) -> Result[User, NotFound]:
    if user_id <= 0:
        return Err(NotFound())
    return Ok(User(id=user_id, name="Alice"))


# ⚠ Explicit type annotation is required — Python's type system cannot infer
# error types through generator expressions. Type checkers see
# Result[str, Never] instead of Result[str, NotFound].
#
# Worse: the annotation is NOT checked. Writing Result[str, int] here
# would silently pass — you lose the type safety that Result exists for.
result: Result[str, NotFound] = do(
    Ok(f"Hello, {user.name}")
    for user in get_user(42)
)

assert result == Ok("Hello, Alice")

# Compare: map() infers the full type automatically — no annotation needed,
# and the error type is checked by the type checker
result2 = get_user(42).map(lambda user: f"Hello, {user.name}")
assert result2 == Ok("Hello, Alice")
```

For async code, use `do_async` (same limitation):

```python
import asyncio
from dataclasses import dataclass
from corrode import do_async, Ok, Err, Result


@dataclass
class FetchError:
    pass


async def fetch_name(user_id: int) -> Result[str, FetchError]:
    return Ok("Alice")


async def main() -> None:
    # ⚠ Same problem: annotation required, not checked, type safety lost
    result: Result[str, FetchError] = await do_async(
        Ok(f"Hello, {name}")
        for name in await fetch_name(42)
    )
    assert result == Ok("Hello, Alice")


asyncio.run(main())
```

`do_async` accepts both sync and async generators.

### `@as_result` / `@as_async_result`

Wraps a function so that it returns `Ok(value)` on success and `Err(exception)`
on specified exception types. Uncaught exception types propagate normally.

```python
import os
from corrode import as_result, Ok

os.environ["PORT"] = "8080"


@as_result(KeyError, ValueError)
def parse_env(key: str) -> int:
    return int(os.environ[key])


result = parse_env("PORT")  # Result[int, KeyError | ValueError]
assert result == Ok(8080)
```

For async functions:

```python
import asyncio
from corrode import as_async_result, Ok


class FetchError(Exception):
    pass


@as_async_result(FetchError)
async def fetch(url: str) -> bytes:
    return b"response data"


async def main() -> None:
    result = await fetch("https://example.com")
    assert result == Ok(b"response data")


asyncio.run(main())
```

At least one exception type is required — calling `@as_result()` with no
arguments raises `TypeError`.

### Escape hatches

For interop with code that doesn't use `Result`, or when you're absolutely
certain about the variant, these methods provide direct access. Prefer
pattern matching and combinators in most cases.

**Extracting values:**

```python
from corrode import Ok, Err, Result

result_ok: Result[int, str] = Ok(42)
result_err: Result[int, str] = Err("oops")

# .ok() and .err() return Optional
assert result_ok.ok() == 42
assert result_ok.err() is None
assert result_err.ok() is None
assert result_err.err() == "oops"

# Direct property access (use when you know the variant)
assert Ok(42).ok_value == 42
assert Err("oops").err_value == "oops"
```

**Unwrapping (raises on wrong variant):**

```python
from corrode import Ok, Err, UnwrapError

# Get value or raise UnwrapError
assert Ok(42).unwrap() == 42
assert Ok(42).expect("should have user") == 42
# Err("oops").unwrap()  # raises UnwrapError

# Get value or use default
assert Ok(42).unwrap_or(0) == 42
assert Err("oops").unwrap_or(0) == 0

# Get value or compute from error
def error_len(e: str) -> int:
    return len(e)


assert Err("oops").unwrap_or_else(error_len) == 4

# Get value or raise custom exception
assert Ok(42).unwrap_or_raise(ValueError) == 42
# Err("oops").unwrap_or_raise(ValueError)  # raises ValueError("oops")
```

**Type guards (for if/else instead of match):**

```python
from corrode import Ok, Err, Result, is_ok, is_err

result: Result[int, str] = Ok(42)

if is_ok(result):
    # Type checker knows result is Ok here
    print(result.ok_value)
elif is_err(result):
    # Type checker knows result is Err here
    print(result.err_value)
```

## Iterator utilities

Functions for working with iterables of `Result` values:

```python
from corrode.iterator import collect, map_collect, partition, filter_ok, filter_err, try_reduce
```

### `collect`

Collect an iterable of `Result` values into `Ok[list]`. Returns the first
`Err` encountered, short-circuiting the iteration:

```python
from corrode import Ok, Err, Result
from corrode.iterator import collect

results: list[Result[int, str]] = [Ok(1), Ok(2), Ok(3)]
assert collect(results) == Ok([1, 2, 3])

results_with_err: list[Result[int, str]] = [Ok(1), Err("bad"), Ok(3)]
assert collect(results_with_err) == Err("bad")
```

### `map_collect`

Apply a function to each element and collect into `Ok[list]`. Returns the
first `Err` produced, short-circuiting the iteration:

```python
from corrode import Ok, Err, Result
from corrode.iterator import map_collect


def parse(s: str) -> Result[int, str]:
    if s.isdigit():
        return Ok(int(s))
    return Err(f"not a number: {s!r}")


assert map_collect(["1", "2", "3"], parse) == Ok([1, 2, 3])
assert map_collect(["1", "x", "3"], parse) == Err("not a number: 'x'")
```

### `partition`

Split an iterable of `Result` into `(oks, errs)`. Consumes all elements
without short-circuiting:

```python
from corrode import Ok, Err, Result
from corrode.iterator import partition

results: list[Result[int, str]] = [Ok(1), Err("a"), Ok(2), Err("b")]
oks, errs = partition(results)
assert oks == [1, 2]
assert errs == ["a", "b"]
```

### `filter_ok`

Yield the value from each `Ok`, skipping `Err` values:

```python
from corrode import Ok, Err, Result
from corrode.iterator import filter_ok

results: list[Result[int, str]] = [Ok(1), Err("x"), Ok(2)]
assert list(filter_ok(results)) == [1, 2]
```

### `filter_err`

Yield the error from each `Err`, skipping `Ok` values:

```python
from corrode import Ok, Err, Result
from corrode.iterator import filter_err

results: list[Result[int, str]] = [Ok(1), Err("x"), Ok(2), Err("y")]
assert list(filter_err(results)) == ["x", "y"]
```

### `try_reduce`

Fold an iterable with a fallible function, short-circuiting on `Err`:

```python
from corrode import Ok, Err, Result
from corrode.iterator import try_reduce


def safe_add(acc: int, x: int) -> Result[int, str]:
    if x < 0:
        return Err(f"negative value: {x}")
    return Ok(acc + x)


assert try_reduce([1, 2, 3], 0, safe_add) == Ok(6)
assert try_reduce([1, -1, 3], 0, safe_add) == Err("negative value: -1")
```

## Async iterator utilities

Functions for concurrent processing of awaitables that return `Result`:

```python
from corrode.async_iterator import (
    collect,
    map_collect,
    partition,
    filter_ok_unordered,
    filter_err_unordered,
    filter_ok,
    filter_err,
    try_reduce,
)
```

All functions accept an optional `concurrency` parameter to limit how many
tasks run at the same time. `None` (default) means unlimited.

`collect`, `map_collect`, `partition`, `filter_ok`, and `filter_err` return
results in **input order**. `filter_ok_unordered` and `filter_err_unordered`
yield in **completion order** (faster, but unordered).

### `collect`

Await an iterable of coroutines or tasks concurrently, collecting results
into `Ok[list]` in input order. Returns the first `Err` encountered, cancelling remaining tasks:

```python
import asyncio
from dataclasses import dataclass
from corrode import Ok, Err, Result
from corrode.async_iterator import collect


@dataclass
class User:
    id: int


@dataclass
class NotFound:
    user_id: int


async def fetch_user(user_id: int) -> Result[User, NotFound]:
    if user_id <= 0:
        return Err(NotFound(user_id=user_id))
    return Ok(User(id=user_id))


async def main() -> None:
    # Results are in input order regardless of completion order
    result = await collect([fetch_user(1), fetch_user(2), fetch_user(3)])
    assert result == Ok([User(id=1), User(id=2), User(id=3)])

    # With concurrency limit — order still matches input
    result = await collect([fetch_user(i) for i in range(1, 6)], concurrency=3)
    assert result == Ok([User(id=1), User(id=2), User(id=3), User(id=4), User(id=5)])


asyncio.run(main())
```

### `map_collect`

Apply an async function to each element concurrently and collect into `Ok[list]`.
Returns the first `Err` produced, cancelling remaining tasks:

```python
import asyncio
from dataclasses import dataclass
from corrode import Ok, Err, Result
from corrode.async_iterator import map_collect


@dataclass
class User:
    id: int


@dataclass
class NotFound:
    user_id: int


async def fetch_user(user_id: int) -> Result[User, NotFound]:
    if user_id <= 0:
        return Err(NotFound(user_id=user_id))
    return Ok(User(id=user_id))


async def main() -> None:
    user_ids = [1, 2, 3, 4, 5]

    # Results are in input order regardless of completion order
    result = await map_collect(user_ids, fetch_user)
    assert result == Ok([User(id=1), User(id=2), User(id=3), User(id=4), User(id=5)])

    # Limit concurrency — order still matches input
    result = await map_collect(user_ids, fetch_user, concurrency=2)
    assert result == Ok([User(id=1), User(id=2), User(id=3), User(id=4), User(id=5)])


asyncio.run(main())
```

### `partition`

Await an iterable of coroutines or tasks concurrently, splitting results into
`(oks, errs)` in input order. Unlike `collect`, never short-circuits — all
awaitables run to completion:

```python
import asyncio
from dataclasses import dataclass
from corrode import Ok, Err, Result
from corrode.async_iterator import partition


@dataclass
class User:
    id: int


@dataclass
class NotFound:
    user_id: int


async def fetch_user(user_id: int) -> Result[User, NotFound]:
    if user_id <= 0:
        return Err(NotFound(user_id=user_id))
    return Ok(User(id=user_id))


async def main() -> None:
    # oks and errs preserve relative input order
    oks, errs = await partition([
        fetch_user(1),
        fetch_user(-1),  # will fail
        fetch_user(2),
        fetch_user(-2),  # will fail
        fetch_user(3),
    ])
    assert oks == [User(id=1), User(id=2), User(id=3)]
    assert errs == [NotFound(user_id=-1), NotFound(user_id=-2)]

    # With concurrency limit — order still matches input
    oks, errs = await partition(
        [fetch_user(i) for i in range(-2, 5)],
        concurrency=3,
    )
    assert oks == [User(id=1), User(id=2), User(id=3), User(id=4)]
    assert errs == [NotFound(user_id=-2), NotFound(user_id=-1), NotFound(user_id=0)]


asyncio.run(main())
```

### `filter_ok_unordered`

Await coroutines or tasks concurrently, yielding `Ok` values as they complete.
`Err` values are silently skipped:

```python
import asyncio
from dataclasses import dataclass
from corrode import Ok, Err, Result
from corrode.async_iterator import filter_ok_unordered


@dataclass
class User:
    id: int
    name: str


@dataclass
class NotFound:
    user_id: int


async def fetch_user(user_id: int) -> Result[User, NotFound]:
    if user_id <= 0:
        return Err(NotFound(user_id=user_id))
    return Ok(User(id=user_id, name=f"User{user_id}"))


async def main() -> None:
    users = []
    async for user in filter_ok_unordered([fetch_user(1), fetch_user(-1), fetch_user(2)]):
        users.append(user)
    assert len(users) == 2

    # With concurrency limit
    users = []
    async for user in filter_ok_unordered(
        [fetch_user(i) for i in range(-2, 5)],
        concurrency=2,
    ):
        users.append(user)
    assert len(users) == 4


asyncio.run(main())
```

### `filter_err_unordered`

Await coroutines or tasks concurrently, yielding `Err` values as they complete.
`Ok` values are silently skipped:

```python
import asyncio
from dataclasses import dataclass
from corrode import Ok, Err, Result
from corrode.async_iterator import filter_err_unordered


@dataclass
class User:
    id: int


@dataclass
class NotFound:
    user_id: int


async def fetch_user(user_id: int) -> Result[User, NotFound]:
    if user_id <= 0:
        return Err(NotFound(user_id=user_id))
    return Ok(User(id=user_id))


async def main() -> None:
    errors = []
    async for err in filter_err_unordered([fetch_user(1), fetch_user(-1), fetch_user(2)]):
        errors.append(err)
    assert errors == [NotFound(user_id=-1)]


asyncio.run(main())
```

### `filter_ok`

Await coroutines or tasks concurrently, yielding `Ok` values in **input order**.
`Err` values are silently skipped. Later-completing tasks are buffered until
all earlier ones have been yielded.

Unlike `filter_ok_unordered`, `concurrency` is required and cannot be `None`
because the reorder buffer would otherwise be unbounded:

```python
import asyncio
from dataclasses import dataclass
from corrode import Ok, Err, Result
from corrode.async_iterator import filter_ok


@dataclass
class User:
    id: int
    name: str


@dataclass
class NotFound:
    user_id: int


async def fetch_user(user_id: int) -> Result[User, NotFound]:
    if user_id <= 0:
        return Err(NotFound(user_id=user_id))
    return Ok(User(id=user_id, name=f"User{user_id}"))


async def main() -> None:
    # Errors are skipped, successes come out in input order
    users = [
        user async for user in filter_ok(
            [fetch_user(1), fetch_user(-1), fetch_user(2), fetch_user(3)],
            concurrency=4,
        )
    ]
    assert users == [User(id=1, name="User1"), User(id=2, name="User2"), User(id=3, name="User3")]


asyncio.run(main())
```

### `filter_err`

Await coroutines or tasks concurrently, yielding `Err` values in **input order**.
`Ok` values are silently skipped. Like `filter_ok`, requires an explicit `concurrency`:

```python
import asyncio
from dataclasses import dataclass
from corrode import Ok, Err, Result
from corrode.async_iterator import filter_err


@dataclass
class User:
    id: int


@dataclass
class NotFound:
    user_id: int


async def fetch_user(user_id: int) -> Result[User, NotFound]:
    if user_id <= 0:
        return Err(NotFound(user_id=user_id))
    return Ok(User(id=user_id))


async def main() -> None:
    errors = [
        err async for err in filter_err(
            [fetch_user(1), fetch_user(-1), fetch_user(2), fetch_user(-2)],
            concurrency=4,
        )
    ]
    # Errors preserve relative input order: -1 before -2
    assert errors == [NotFound(user_id=-1), NotFound(user_id=-2)]


asyncio.run(main())
```

### `try_reduce`

Await each coroutine or task **sequentially**, folding results with a fallible
function. Short-circuits on the first `Err` and closes remaining coroutines.

Unlike `collect` / `partition`, tasks run one at a time because each awaited
value must be passed to the accumulator before the next task can start:

```python
import asyncio
from corrode import Ok, Err, Result
from corrode.async_iterator import try_reduce


async def fetch_price(item_id: int) -> int:
    prices = {1: 100, 2: 250, 3: 75}
    return prices.get(item_id, -1)


def accumulate(total: int, price: int) -> Result[int, str]:
    if price < 0:
        return Err(f"unknown item with price {price}")
    return Ok(total + price)


async def main() -> None:
    result = await try_reduce(
        [fetch_price(1), fetch_price(2), fetch_price(3)],
        initial=0,
        f=accumulate,
    )
    assert result == Ok(425)

    # Short-circuits on the first Err
    result = await try_reduce(
        [fetch_price(1), fetch_price(99), fetch_price(3)],
        initial=0,
        f=accumulate,
    )
    assert result == Err("unknown item with price -1")


asyncio.run(main())
```

## Acknowledgements

`corrode` is inspired by and originally forked from [rustedpy/result](https://github.com/rustedpy/result). 
We are grateful for that library's existence — it laid the foundation for bringing Rust-style result types to Python and made this project possible.

## License

MIT License
