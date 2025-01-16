"""Data model, registration, and parametrization facilities for defining benchmarks."""

import inspect
import itertools
import sys
import types
import warnings
from collections.abc import Callable, Iterable
from typing import Any, Union, get_args, get_origin, overload

from nnbench.memo import is_memo, is_memo_type
from nnbench.types import Benchmark, NoOp


def _check_against_interface(params: dict[str, Any], fun: Callable) -> None:
    sig = inspect.signature(fun)
    fvarnames = set(sig.parameters.keys())
    fvartypes = {k: v.annotation for k, v in sig.parameters.items()}
    varnames = set(params.keys())
    if not varnames <= fvarnames:
        # never empty due to the if condition.
        diffvar, *_ = varnames - fvarnames
        raise TypeError(
            f"benchmark {fun.__name__}() got an unexpected keyword argument {diffvar!r}"
        )
    # at this point, params.keys() <= argnames.
    for k, v in params.items():
        fvtype = fvartypes[k]
        # if no type annotation is given, everything is allowed.
        if fvtype == inspect.Parameter.empty:
            continue
        # to unwrap generic containers like list[str].
        expected_type = get_origin(fvtype) or fvtype
        # in case of a union like str | int, check args instead.
        union_type = Union if sys.version_info < (3, 10) else types.UnionType
        if expected_type is union_type:
            expected_type = get_args(fvtype)
        if is_memo(v) and not is_memo_type(fvtype):
            expected_type = inspect.signature(v).return_annotation
            # in the memo case, we can compare only types, not values.
            if expected_type == fvtype:
                continue
        if not isinstance(v, expected_type):
            raise TypeError(
                f"benchmark {fun.__name__}(): expected type {fvtype}, "
                f"got type {type(v)} for parametrized argument {k!r}"
            )


def _default_namegen(fn: Callable, **kwargs: Any) -> str:
    return fn.__name__ + "_" + "_".join(f"{k}={v}" for k, v in kwargs.items())


# Overloads for the ``benchmark`` decorator.
# Case #1: Bare application without parentheses
# @nnbench.benchmark
# def foo() -> int:
#     return 0
@overload
def benchmark(
    func: None = None,
    name: str = "",
    setUp: Callable[..., None] = NoOp,
    tearDown: Callable[..., None] = NoOp,
    tags: tuple[str, ...] = (),
) -> Callable[[Callable], Benchmark]: ...


# Case #2: Application with arguments
# @nnbench.benchmark(name="My foo experiment", tags=("hello", "world"))
# def foo() -> int:
#     return 0
@overload
def benchmark(
    func: Callable[..., Any],
    name: str = "",
    setUp: Callable[..., None] = NoOp,
    tearDown: Callable[..., None] = NoOp,
    tags: tuple[str, ...] = (),
) -> Benchmark: ...


def benchmark(
    func: Callable[..., Any] | None = None,
    name: str = "",
    setUp: Callable[..., None] = NoOp,
    tearDown: Callable[..., None] = NoOp,
    tags: tuple[str, ...] = (),
) -> Benchmark | Callable[[Callable], Benchmark]:
    """
    Define a benchmark from a function.

    The resulting benchmark can either be completely (i.e., the resulting function takes no
    more arguments) or incompletely parametrized. In the latter case, the remaining free
    parameters need to be passed in the calls to `BenchmarkRunner.run()`.

    Parameters
    ----------
    func: Callable[..., Any] | None
        The function to benchmark. This slot only exists to allow application of the decorator
        without parentheses, you should never fill it explicitly.
    name: str
        A display name to give to the benchmark. Useful in summaries and reports.
    setUp: Callable[..., None]
        A setup hook to run before the benchmark.
    tearDown: Callable[..., None]
        A teardown hook to run after the benchmark.
    tags: tuple[str, ...]
        Additional tags to attach for bookkeeping and selective filtering during runs.

    Returns
    -------
    Benchmark | Callable[[Callable], Benchmark]
        The resulting benchmark (if no arguments were given), or a parametrized decorator
        returning the benchmark.
    """

    def decorator(fun: Callable) -> Benchmark:
        return Benchmark(fun, name=name, setUp=setUp, tearDown=tearDown, tags=tags)

    if func is not None:
        return decorator(func)
    else:
        return decorator


def parametrize(
    parameters: Iterable[dict[str, Any]],
    setUp: Callable[..., None] = NoOp,
    tearDown: Callable[..., None] = NoOp,
    namegen: Callable[..., str] = _default_namegen,
    tags: tuple[str, ...] = (),
) -> Callable[[Callable], list[Benchmark]]:
    """
    Define a family of benchmarks over a function with varying parameters.

    The resulting benchmarks can either be completely (i.e., the resulting function takes no
    more arguments) or incompletely parametrized. In the latter case, the remaining free
    parameters need to be passed in the call to `BenchmarkRunner.run()`.

    Parameters
    ----------
    parameters: Iterable[dict[str, Any]]
        The different sets of parameters defining the benchmark family.
    setUp: Callable[..., None]
        A setup hook to run before each of the benchmarks.
    tearDown: Callable[..., None]
        A teardown hook to run after each of the benchmarks.
    namegen: Callable[..., str]
        A function taking the benchmark function and given parameters that generates a unique
        custom name for the benchmark. The default name generated is the benchmark function's name
        followed by the keyword arguments in ``key=value`` format separated by underscores.
    tags: tuple[str, ...]
        Additional tags to attach for bookkeeping and selective filtering during runs.

    Returns
    -------
    Callable[[Callable], list[Benchmark]]
        A parametrized decorator returning the benchmark family.
    """

    # TODO(nicholasjng): Return a BenchmarkFamily object.
    def decorator(fn: Callable) -> list[Benchmark]:
        benchmarks = []
        names = set()
        for params in parameters:
            _check_against_interface(params, fn)

            name = namegen(fn, **params)
            if name in names:
                warnings.warn(
                    f"Got duplicate name {name!r} for benchmark {fn.__name__}(). "
                    f"Perhaps you specified a parameter configuration twice?"
                )
            names.add(name)

            bm = Benchmark(
                fn,
                name=name,
                params=params,
                setUp=setUp,
                tearDown=tearDown,
                tags=tags,
            )
            benchmarks.append(bm)
        return benchmarks

    return decorator


def product(
    setUp: Callable[..., None] = NoOp,
    tearDown: Callable[..., None] = NoOp,
    namegen: Callable[..., str] = _default_namegen,
    tags: tuple[str, ...] = (),
    **iterables: Iterable,
) -> Callable[[Callable], list[Benchmark]]:
    """
    Define a family of benchmarks over a cartesian product of one or more iterables.

    The resulting benchmarks can either be completely (i.e., the resulting function takes no
    more arguments) or incompletely parametrized. In the latter case, the remaining free
    parameters need to be passed in the call to `BenchmarkRunner.run()`.

    Parameters
    ----------
    setUp: Callable[..., None]
        A setup hook to run before each of the benchmarks.
    tearDown: Callable[..., None]
        A teardown hook to run after each of the benchmarks.
    namegen: Callable[..., str]
        A function taking the benchmark function and given parameters that generates a unique
        custom name for the benchmark. The default name generated is the benchmark function's name
        followed by the keyword arguments in ``key=value`` format separated by underscores.
    tags: tuple[str, ...]
        Additional tags to attach for bookkeeping and selective filtering during runs.
    **iterables: Iterable
        The iterables parametrizing the benchmarks.

    Returns
    -------
    Callable[[Callable], list[Benchmark]]
        A parametrized decorator returning the benchmark family.
    """

    # TODO(nicholasjng): Return a BenchmarkFamily object.
    def decorator(fn: Callable) -> list[Benchmark]:
        benchmarks = []
        names = set()
        varnames = iterables.keys()
        for values in itertools.product(*iterables.values()):
            params = dict(zip(varnames, values))
            _check_against_interface(params, fn)

            name = namegen(fn, **params)
            if name in names:
                warnings.warn(
                    f"Got duplicate name {name!r} for benchmark {fn.__name__}(). "
                    f"Perhaps you specified a parameter configuration twice?"
                )
            names.add(name)

            bm = Benchmark(
                fn,
                name=name,
                params=params,
                setUp=setUp,
                tearDown=tearDown,
                tags=tags,
            )
            benchmarks.append(bm)
        return benchmarks

    return decorator
