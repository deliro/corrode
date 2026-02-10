# corrode

[![CI](https://github.com/deliro/corrode/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/deliro/corrode/actions/workflows/ci.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/deliro/corrode/branch/main/graph/badge.svg)](https://codecov.io/gh/deliro/corrode)

A Rust-like `Result` type for Python 3.11+, fully type annotated.

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
def get_user(user_id: int) -> User:
    # Can this raise? What exceptions? The caller has no idea.
    ...
```

`Result` makes errors explicit, typed, and impossible to ignore:

```python
def get_user(user_id: int) -> Result[User, NotFound | Forbidden]:
    ...
```

Now every caller sees the possible errors in the signature, the type checker
verifies every branch is handled, and adding a new error variant is
a compile-time breaking change — not a runtime surprise.

## Quick start

`Result[T, E]` is a union of `Ok[T] | Err[E]`. Every `Result` must be explicitly
handled — no silent `None`s, no uncaught exceptions.

```python
from dataclasses import dataclass
from corrode import Ok, Err, Result


@dataclass
class User:
    id: int
    name: str
    email: str


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
    return Ok(User(id=user_id, name="Alice", email="alice@example.com"))
```

## Exhaustive error handling

Use a nested `match` on the error value together with `assert_never` to get
a compile-time guarantee that every error variant is handled:

```python
from typing import assert_never

match get_user(42):
    case Ok(user):
        print(f"Welcome, {user.name}")
    case Err(e):
        match e:
            case NotFound(user_id=uid):
                print(f"User {uid} does not exist")
            case Forbidden(reason=reason):
                print(f"Access denied: {reason}")
            case _:
                assert_never(e)
```

Now add a new error variant:

```python
@dataclass
class RateLimited:
    retry_after: float

type GetUserError = NotFound | Forbidden | RateLimited
```

Without changing anything else, `mypy` immediately reports:

```
error: Argument 1 to "assert_never" has incompatible type "RateLimited"; expected "Never"
```

You are forced to handle the new case before the code passes type checking.
No error silently slips through.

## Adopting corrode in an existing codebase

You don't have to rewrite everything at once. Exceptions don't disappear
overnight, and third-party libraries will always raise them. That's fine —
`corrode` is designed for gradual adoption.

### Step 1: wrap existing functions with `@as_result`

You have code that raises. Don't rewrite it yet — just wrap it:

```python
from corrode import as_result

# Before: raises KeyError, ValueError, nobody knows about it
def parse_port(key: str) -> int:
    return int(os.environ[key])

# After: signature tells you exactly what can go wrong
@as_result(KeyError, ValueError)
def parse_port(key: str) -> int:
    return int(os.environ[key])
```

The function body stays the same. The only change is the decorator, and
the callers now get a `Result` instead of praying nothing blows up:

```python
from typing import assert_never

match parse_port("PORT"):
    case Ok(port):
        start_server(port)
    case Err(e):
        match e:
            case KeyError():
                start_server(8080)
            case ValueError():
                sys.exit(f"Invalid PORT: {e}")
            case _:
                assert_never(e)
```

### Step 2: return `Err(exception)` explicitly

Once callers are adapted, you can drop the decorator and return errors
explicitly. The function still uses exception classes, so the callers
don't change:

```python
def parse_port(key: str) -> Result[int, KeyError | ValueError]:
    raw = os.environ.get(key)
    if raw is None:
        return Err(KeyError(key))
    try:
        return Ok(int(raw))
    except ValueError as exc:
        return Err(exc)
```

### Step 3: replace exceptions with domain types

When you're ready, replace exception classes with dataclasses that carry
exactly the data the caller needs:

```python
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
```

Each step is a small, safe refactoring. Your callers get progressively
better types, and `mypy` catches every unhandled case.

### Exceptions inside Result-returning code

Third-party libraries raise exceptions — that's fine. A `try/except`
inside a function that returns `Result` is completely normal:

```python
import httpx


def fetch_data(url: str) -> Result[bytes, NotFound | Unavailable]:
    try:
        response = httpx.get(url)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return Err(NotFound(url=url))
        return Err(Unavailable(url=url, status=exc.response.status_code))
    except httpx.ConnectError:
        return Err(Unavailable(url=url, status=0))
    return Ok(response.content)
```

You catch the exception, convert it to a typed `Err` with exactly the
data the caller needs, and the rest of your code stays in `Result`-land.
No need to wrap every library call — just handle exceptions where they
happen and return a meaningful error.

If a function simply re-raises a third-party exception without
transformation, `@as_result` can save some boilerplate:

```python
from corrode import as_result

@as_result(httpx.ConnectError)
def fetch_bytes(url: str) -> bytes:
    return httpx.get(url).content
```

## API reference

### Pattern matching

The preferred way to handle results — see
[Exhaustive error handling](#exhaustive-error-handling) above for the full
pattern with `assert_never`.

`Ok` and `Err` support structural pattern matching via `__match_args__`:

```python
match get_user(42):
    case Ok(user):
        print(user.name)
    case Err(error):
        print(error)
```

### Checking and narrowing

Type guard functions narrow the type so the type checker knows exactly
which variant you have:

```python
from corrode import is_ok, is_err

result = get_user(42)

if is_ok(result):
    # type checker sees Ok[User] here
    send_email(result.ok_value)
elif is_err(result):
    # type checker sees Err[GetUserError] here
    log(result.err_value)
```

Methods on the result itself:

```python
Ok(1).is_ok()    # True
Ok(1).is_err()   # False
Err(1).is_ok()   # False
Err(1).is_err()  # True
```

### Accessors

Extract the inner value as an `Optional`:

```python
Ok(1).ok()       # 1
Ok(1).err()      # None

Err("e").ok()    # None
Err("e").err()   # "e"
```

Direct access via properties:

```python
Ok(1).ok_value       # 1
Err("e").err_value   # "e"
```

### Unwrapping

Extract the value when you're certain it's `Ok`, or provide a fallback:

```python
Ok(1).unwrap()                           # 1
Ok(1).expect("must be present")          # 1

Err("e").unwrap()                        # raises UnwrapError
Err("e").expect("must be present")       # raises UnwrapError

Ok(1).unwrap_or(0)                       # 1
Err("e").unwrap_or(0)                    # 0

Ok(1).unwrap_or_else(lambda e: 0)        # 1
Err("e").unwrap_or_else(str.upper)       # "E"

Ok(1).unwrap_or_raise(ValueError)        # 1
Err("e").unwrap_or_raise(ValueError)     # raises ValueError("e")
```

The reverse — extract the error:

```python
Err("e").unwrap_err()                    # "e"
Err("e").expect_err("must be err")       # "e"

Ok(1).unwrap_err()                       # raises UnwrapError
Ok(1).expect_err("must be err")          # raises UnwrapError
```

`UnwrapError` carries the original result:

```python
try:
    Err("e").unwrap()
except UnwrapError as exc:
    exc.result  # Err("e")
```

If the contained error is a `BaseException`, it is chained as `__cause__`:

```python
try:
    Err(ValueError("bad")).unwrap()
except UnwrapError as exc:
    exc.__cause__  # ValueError("bad")
```

Async variant: `unwrap_or_else_async`.

### Transforming values

```python
Ok(2).map(lambda x: x * 3)              # Ok(6)
Err("e").map(lambda x: x * 3)           # Err("e") — untouched

Err(404).map_err(lambda c: f"HTTP {c}")  # Err("HTTP 404")
Ok(1).map_err(lambda c: f"HTTP {c}")     # Ok(1) — untouched

Ok(1).map_or(-1, lambda x: x + 1)       # 2
Err(1).map_or(-1, lambda x: x + 1)      # -1

Ok(1).map_or_else(lambda e: -1, lambda x: x + 1)       # 2
Err("e").map_or_else(lambda e: e.upper(), lambda x: x)  # "E"
```

Async variants: `map_async`, `map_err_async`, `map_or_async`,
`map_or_else_async`.

### Chaining with `and_then` / `or_else`

`and_then` calls the function only if `Ok`, otherwise forwards the `Err`:

```python
def parse_int(s: str) -> Result[int, str]:
    if s.isdigit():
        return Ok(int(s))
    return Err(f"not a number: {s!r}")

def check_positive(n: int) -> Result[int, str]:
    if n > 0:
        return Ok(n)
    return Err(f"{n} is not positive")

parse_int("42").and_then(check_positive)   # Ok(42)
parse_int("-1").and_then(check_positive)   # Err("-1 is not positive")
parse_int("abc").and_then(check_positive)  # Err("not a number: 'abc'")
```

`or_else` calls the function only if `Err`, otherwise forwards the `Ok`:

```python
Ok(2).or_else(lambda e: Ok(0))           # Ok(2) — untouched
Err(3).or_else(lambda e: Ok(e * e))      # Ok(9)
Err(3).or_else(lambda e: Err(e))         # Err(3)
```

Async variants: `and_then_async`, `or_else_async`.

### Predicates

```python
Ok(4).is_ok_and(lambda x: x > 2)         # True
Ok(0).is_ok_and(lambda x: x > 2)         # False
Err("e").is_ok_and(lambda x: x > 2)      # False

Err(404).is_err_and(lambda e: e >= 400)   # True
Err(200).is_err_and(lambda e: e >= 400)   # False
Ok(1).is_err_and(lambda e: e >= 400)      # False
```

Async variants: `is_ok_and_async`, `is_err_and_async`.

### Inspecting

Call a function on the contained value for side effects without consuming
the result:

```python
Ok(42).inspect(print)           # prints 42, returns Ok(42)
Err("e").inspect(print)         # does nothing, returns Err("e")

Err("e").inspect_err(print)     # prints "e", returns Err("e")
Ok(42).inspect_err(print)       # does nothing, returns Ok(42)
```

Async variants: `inspect_async`, `inspect_err_async`.

### Async methods

Every transformation and predicate method has an `_async` variant that
accepts an async callable and returns an awaitable:

```python
async def fetch_name(user_id: int) -> str:
    ...

result: Result[int, str] = Ok(42)

# map_async
ok = await result.map_async(fetch_name)

# and_then_async
async def validate(x: int) -> Result[int, str]:
    ...

ok = await result.and_then_async(validate)

# unwrap_or_else_async
async def fallback(e: str) -> int:
    return 0

value = await result.unwrap_or_else_async(fallback)
```

Full list: `map_async`, `map_err_async`, `map_or_async`,
`map_or_else_async`, `and_then_async`, `or_else_async`,
`unwrap_or_else_async`, `is_ok_and_async`, `is_err_and_async`,
`inspect_async`, `inspect_err_async`.

### `do` notation

Syntactic sugar for a sequence of `and_then()` calls. If any step is `Err`,
the whole expression short-circuits:

```python
from corrode import do, Ok, Result


def get_subscription(user: User) -> Result[Subscription, GetUserError]: ...

result: Result[str, GetUserError] = do(
    Ok(f"{user.name} has {sub.plan}")
    for user in get_user(42)
    for sub in get_subscription(user)
)
```

For async code, use `do_async`:

```python
from corrode import do_async

result: Result[str, FetchError] = await do_async(
    Ok(f"{user.name}: {profile.bio}")
    for user in await fetch_user(42)
    for profile in await fetch_profile(user.id)
)
```

`do_async` accepts both sync and async generators.

### `@as_result` / `@as_async_result`

Wraps a function so that it returns `Ok(value)` on success and `Err(exception)`
on specified exception types. Uncaught exception types propagate normally.

```python
@as_result(KeyError, ValueError)
def parse_env(key: str) -> int:
    return int(os.environ[key])

parse_env("PORT")  # Result[int, KeyError | ValueError]
```

For async functions:

```python
@as_async_result(httpx.HTTPError)
async def fetch(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content

await fetch("https://example.com")  # Result[bytes, httpx.HTTPError]
```

At least one exception type is required — calling `@as_result()` with no
arguments raises `TypeError`.

## License

MIT License
