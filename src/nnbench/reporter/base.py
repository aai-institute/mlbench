from __future__ import annotations

import re
from typing import Any, Callable, Sequence

from tabulate import tabulate

from nnbench.context import Context
from nnbench.reporter.util import nullcols
from nnbench.types import BenchmarkRecord


# TODO: Add IO mixins for database, file, and HTTP IO
class BenchmarkReporter:
    """
    The base interface for a benchmark reporter class.

    A benchmark reporter consumes benchmark results from a previous run, and subsequently
    reports them in the way specified by the respective implementation's ``report_result()``
    method.

    For example, to write benchmark results to a database, you could save the credentials
    for authentication on the class, and then stream the results directly to
    the database in ``report_result()``, with preprocessing if necessary.

    Parameters
    ----------
    tablefmt: str
        A table format identifier to use when displaying records in the console.
    custom_formatters: dict[str, Callable[[Any], Any]] | None
        A mapping of column names to custom formatters, i.e. functions formatting input
        values for display in the console.
    """

    def __init__(
        self,
        tablefmt: str = "simple",
        custom_formatters: dict[str, Callable[[Any], Any]] | None = None,
    ):
        self.tablefmt = tablefmt
        self.custom_formatters: dict[str, Callable[[Any], Any]] = custom_formatters or {}

    def initialize(self):
        """
        Initialize the reporter's state.

        This is the intended place to create resources like a result directory,
        a database connection, or a HTTP client.
        """
        pass

    def finalize(self):
        """
        Finalize the reporter's state.

        This is the intended place to destroy/release resources that were previously
        acquired in ``initialize()``.
        """
        pass

    def display(
        self,
        record: BenchmarkRecord,
        benchmark_filter: str | None = None,
        include: tuple[str, ...] | None = None,
        exclude: tuple[str, ...] = (),
        include_context: tuple[str, ...] = (),
        exclude_empty: bool = True,
    ) -> None:
        """
        Display a benchmark record in the console.

        Benchmarks and context values will be filtered before display
        if any filtering is applied.

        Columns that do not contain any useful information are omitted by default.

        Parameters
        ----------
        record: BenchmarkRecord
            The benchmark record to display.
        benchmark_filter: str | None
            A regex used to match benchmark names whose results to display.
        include: tuple[str, ...] | None
            Columns to include in the displayed table.
        exclude: tuple[str, ...]
            Columns to exclude from the displayed table.
        include_context: tuple[str, ...]
            Context values to include. Supports nested attribute via dot syntax,
            i.e. a name "foo.bar" causes the member ``"bar"`` of the context value
            ``"foo"`` to be displayed.
        exclude_empty: bool
            Exclude columns that only contain false-ish values.
        """
        ctx, benchmarks = record["context"], record["benchmarks"]

        # This assumes a stable schema across benchmarks.
        if include is None:
            includes = set(benchmarks[0].keys())
        else:
            includes = set(include)

        excludes = set(exclude)
        nulls = set() if not exclude_empty else nullcols(benchmarks)
        cols = includes - nulls - excludes

        if benchmark_filter is not None:
            regex = re.compile(benchmark_filter, flags=re.IGNORECASE)
        else:
            regex = None

        filtered = []
        for bm in benchmarks:
            if regex is not None and regex.search(bm["name"]) is None:
                continue
            filteredctx = {
                k: v
                for k, v in Context.flatten_dict(ctx).items()
                if any(k.startswith(i) for i in include_context)
            }
            filteredbm = {k: v for k, v in bm.items() if k in cols}
            filteredbm.update(filteredctx)
            # only apply custom formatters after context merge
            #  to allow custom formatting of context values.
            filteredbm = {
                k: self.custom_formatters.get(k, lambda x: x)(v) for k, v in filteredbm.items()
            }
            filtered.append(filteredbm)

        print(tabulate(filtered, headers="keys", tablefmt=self.tablefmt))

    def read(self) -> BenchmarkRecord:
        raise NotImplementedError

    def read_batched(self) -> list[BenchmarkRecord]:
        raise NotImplementedError

    def write(self, record: BenchmarkRecord) -> None:
        raise NotImplementedError

    def write_batched(self, records: Sequence[BenchmarkRecord]) -> None:
        # By default, just loop over the records and write() everything.
        for record in records:
            self.write(record)
