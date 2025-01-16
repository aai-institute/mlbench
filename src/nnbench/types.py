"""Types for benchmarks and records holding results of a run."""

import copy
import inspect
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, TypeVar

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


@dataclass(frozen=True)
class State:
    """
    A dataclass holding some basic information about a benchmark and its hierarchy
    inside its *family* (i.e. a series of the same benchmark for different parameters).

    For benchmarks registered with ``@nnbench.benchmark``, meaning no parametrization,
    each benchmark constitutes its own family, and ``family_size == 1`` holds true.
    """

    name: str
    family: str
    family_size: int
    family_index: int


def NoOp(state: State, params: Mapping[str, Any] = MappingProxyType({})) -> None:
    """A no-op setup/teardown callback that does nothing."""
    pass


@dataclass(frozen=True)
class BenchmarkRecord:
    """
    A dataclass representing the result of a benchmark run, i.e. the return value
    of a call to ``BenchmarkRunner.run()``.
    """

    run: str
    """A name describing the run."""
    context: dict[str, Any]
    """A map of key-value pairs describing context information around the benchmark run."""
    benchmarks: list[dict[str, Any]]
    """The list of benchmark results, each given as a Python dictionary."""

    def to_json(self) -> dict[str, Any]:
        """
        Export a benchmark record to JSON.

        Returns
        -------
        dict[str, Any]
            A JSON representation of the benchmark record.
        """
        return asdict(self)

    def to_list(self) -> list[dict[str, Any]]:
        """
        Export a benchmark record to a list of individual results,
        each with the benchmark run name and context inlined.
        """
        results = []
        for b in self.benchmarks:
            bc = copy.deepcopy(b)
            bc["context"] = self.context
            bc["run"] = self.run
            results.append(bc)
        return results

    @classmethod
    def expand(cls, bms: dict[str, Any] | list[dict[str, Any]]) -> Self:
        """
        Expand a list of deserialized JSON-like objects into a benchmark record.
        This is equivalent to extracting the context given by the method it was
        serialized with, and then returning the rest of the data as is.

        Parameters
        ----------
        bms: dict[str, Any] | list[dict[str, Any]]
            The deserialized benchmark record or list of records to expand into a record.

        Returns
        -------
        BenchmarkRecord
            The resulting record, with the context and run name extracted.
        """
        context: dict[str, Any]
        if isinstance(bms, dict):
            if "benchmarks" not in bms.keys():
                raise ValueError(f"no benchmark data found in struct {bms}")

            benchmarks = bms["benchmarks"]
            context = bms.get("context", {})
            run = bms.get("run", "")
        else:
            run = ""
            context = {}
            benchmarks = bms
            for b in benchmarks:
                # TODO(nicholasjng): This does not do the right thing if the list contains
                #  data from multiple benchmark runs, e.g. from a DB query.
                if "run" in b:
                    run = b.pop("run")
                if "context" in b:
                    # TODO: Log context key/value disagreements
                    context |= b.pop("context", {})
        return cls(run=run, benchmarks=benchmarks, context=context)


@dataclass(init=False, frozen=True)
class Parameters:
    """
    A dataclass designed to hold benchmark parameters.

    This class is not functional on its own, and needs to be subclassed
    according to your benchmarking workloads.

    The main advantage over passing parameters as a dictionary are static analysis
    and type safety for your benchmarking code.
    """


T = TypeVar("T")
Variable = tuple[str, type, Any]


@dataclass(frozen=True)
class Interface:
    """
    Data model representing a function's interface.

    An instance of this class is created using the ``Interface.from_callable()``
    class method.
    """

    funcname: str
    """Name of the function."""
    names: tuple[str, ...]
    """Names of the function parameters."""
    types: tuple[type, ...]
    """Type hints of the function parameters."""
    defaults: tuple
    """The function parameters' default values, or inspect.Parameter.empty if a parameter has no default."""
    variables: tuple[Variable, ...]
    """A tuple of tuples, where each inner tuple contains the parameter name, type, and default value."""
    returntype: type
    """The function's return type annotation, or NoneType if left untyped."""

    @classmethod
    def from_callable(cls, fn: Callable, defaults: dict[str, Any]) -> Self:
        """
        Creates an interface instance from the given callable.

        Wraps the information given by ``inspect.signature()``, with the option to
        supply a ``defaults`` map and overwrite any default set in the function's
        signature.
        """
        # Set `follow_wrapped=False` to get the partially filled interfaces.
        # Otherwise we get missing value errors for parameters supplied in benchmark decorators.
        sig = inspect.signature(fn, follow_wrapped=False)
        ret = sig.return_annotation
        _defaults = {k: defaults.get(k, v.default) for k, v in sig.parameters.items()}
        # defaults are the signature parameters, then the partial parametrization.
        return cls(
            fn.__name__,
            tuple(sig.parameters.keys()),
            tuple(p.annotation for p in sig.parameters.values()),
            tuple(_defaults.values()),
            tuple((k, v.annotation, _defaults[k]) for k, v in sig.parameters.items()),
            type(ret) if ret is None else ret,
        )


@dataclass(frozen=True)
class Benchmark:
    """
    Data model representing a benchmark. Subclass this to define your own custom benchmark.
    """

    fn: Callable[..., Any]
    """The function defining the benchmark."""
    name: str = ""
    """A name to display for the given benchmark. If not given, a name will be constructed from the function name and given parameters."""
    params: dict[str, Any] = field(default_factory=dict)
    """A partial parametrization to apply to the benchmark function. Internal only, you should not need to set this yourself."""
    setUp: Callable[[State, Mapping[str, Any]], None] = field(repr=False, default=NoOp)
    """A setup hook run before the benchmark. Must take all members of ``params`` as inputs."""
    tearDown: Callable[[State, Mapping[str, Any]], None] = field(repr=False, default=NoOp)
    """A teardown hook run after the benchmark. Must take all members of ``params`` as inputs."""
    tags: tuple[str, ...] = field(repr=False, default=())
    """Additional tags to attach for bookkeeping and selective filtering during runs."""
    interface: Interface = field(init=False, repr=False)
    """Benchmark interface, constructed from the given function. Implementation detail."""

    def __post_init__(self):
        if not self.name:
            super().__setattr__("name", self.fn.__name__)
        super().__setattr__("interface", Interface.from_callable(self.fn, self.params))


class BenchmarkFamily(Iterable[Benchmark]):
    def __init__(self, fn: Callable[..., Any], params: Iterable[dict[str, Any]]):
        self.fn = fn
        self.params = params

    def __iter__(self):
        # TODO(nicholasjng): Yield benchmark members as currently done in
        #  `core.{parametrize,product}`.
        pass
