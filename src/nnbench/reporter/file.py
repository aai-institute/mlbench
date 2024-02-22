from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Callable, Literal, Sequence, cast

from nnbench.reporter.base import BenchmarkReporter
from nnbench.types import BenchmarkRecord


@dataclass(frozen=True)
class FileDriverOptions:
    options: dict[str, Any] = field(default_factory=dict)
    """Options to pass to the underlying serializer library call, e.g. ``json.dump``."""
    ctxmode: Literal["flatten", "inline", "omit"] = "inline"
    """How to handle the context struct."""


_Options = dict[str, Any]
SerDe = tuple[
    Callable[[Sequence[BenchmarkRecord], IO, FileDriverOptions], None],
    Callable[[IO, FileDriverOptions], list[BenchmarkRecord]],
]


_file_drivers: dict[str, SerDe] = {}
_compressions: dict[str, Callable] = {}
_file_driver_lock = threading.Lock()
_compression_lock = threading.Lock()


def yaml_save(records: Sequence[BenchmarkRecord], fp: IO, fdoptions: FileDriverOptions) -> None:
    try:
        import yaml
    except ImportError:
        raise ModuleNotFoundError("`pyyaml` is not installed")

    bms = []
    for r in records:
        bms += r.compact(mode=fdoptions.ctxmode)
    yaml.safe_dump(bms, fp, **fdoptions.options)


def yaml_load(fp: IO, fdoptions: FileDriverOptions) -> list[BenchmarkRecord]:
    try:
        import yaml
    except ImportError:
        raise ModuleNotFoundError("`pyyaml` is not installed")

    # TODO: Use expandmany()
    bms = yaml.safe_load(fp)
    return [BenchmarkRecord.expand(bms)]


def json_save(records: Sequence[BenchmarkRecord], fp: IO, fdoptions: FileDriverOptions) -> None:
    import json

    newline: bool = fdoptions.options.pop("newline", False)
    bm = []
    for r in records:
        bm += r.compact(mode=fdoptions.ctxmode)
    if newline:
        fp.write("\n".join([json.dumps(b) for b in bm]))
    else:
        json.dump(bm, fp, **fdoptions.options)


def json_load(fp: IO, fdoptions: FileDriverOptions) -> list[BenchmarkRecord]:
    import json

    newline: bool = fdoptions.options.pop("newline", False)
    benchmarks: list[dict[str, Any]]
    if newline:
        benchmarks = [json.loads(line, **fdoptions.options) for line in fp]
    else:
        benchmarks = json.load(fp, **fdoptions.options)
    return [BenchmarkRecord.expand(benchmarks)]


def csv_save(records: Sequence[BenchmarkRecord], fp: IO, fdoptions: FileDriverOptions) -> None:
    import csv

    bm = []
    for r in records:
        bm += r.compact(mode=fdoptions.ctxmode)
    writer = csv.DictWriter(fp, fieldnames=bm[0].keys(), **fdoptions.options)

    for b in bm:
        writer.writerow(b)


def csv_load(fp: IO, fdoptions: FileDriverOptions) -> list[BenchmarkRecord]:
    import csv

    reader = csv.DictReader(fp, **fdoptions.options)

    benchmarks: list[dict[str, Any]] = []
    # apparently csv.DictReader has no appropriate type hint for __next__,
    # so we supply one ourselves.
    bm: dict[str, Any]
    for bm in reader:
        benchmarks.append(bm)

    return [BenchmarkRecord.expand(benchmarks)]


with _file_driver_lock:
    _file_drivers["yaml"] = (yaml_save, yaml_load)
    _file_drivers["json"] = (json_save, json_load)
    _file_drivers["csv"] = (csv_save, csv_load)
    # TODO: Add parquet support


def get_driver_implementation(name: str) -> SerDe:
    try:
        return _file_drivers[name]
    except KeyError:
        raise KeyError(f"unsupported file format {name!r}") from None


def register_driver_implementation(name: str, impl: SerDe, clobber: bool = False) -> None:
    if name in _file_drivers and not clobber:
        raise RuntimeError(
            f"driver {name!r} is already registered. To force registration, "
            f"rerun with clobber=True"
        )

    with _file_driver_lock:
        _file_drivers[name] = impl


def deregister_driver_implementation(name: str) -> SerDe | None:
    with _file_driver_lock:
        return _file_drivers.pop(name, None)


def gzip_compression(filename: str | os.PathLike[str], mode: Literal["r", "w"] = "r") -> IO:
    import gzip

    # gzip.GzipFile does not inherit from IO[bytes],
    # but it has all required methods, so we allow it.
    return cast(IO[bytes], gzip.GzipFile(filename=filename, mode=mode))


def bz2_compression(filename: str | os.PathLike[str], mode: Literal["r", "w"] = "r") -> IO:
    import bz2

    return bz2.BZ2File(filename=filename, mode=mode)


with _compression_lock:
    _compressions["gzip"] = gzip_compression
    _compressions["bz2"] = bz2_compression


def get_compression_algorithm(name: str) -> Callable:
    try:
        return _compressions[name]
    except KeyError:
        raise KeyError(f"unsupported compression algorithm {name!r}") from None


def register_compression(name: str, impl: Callable, clobber: bool = False) -> None:
    if name in _compressions and not clobber:
        raise RuntimeError(
            f"compression {name!r} is already registered. To force registration, "
            f"rerun with clobber=True"
        )

    with _compression_lock:
        _compressions[name] = impl


def deregister_compression(name: str) -> Callable | None:
    with _compression_lock:
        return _compressions.pop(name, None)


class FileIO:
    def read(
        self,
        file: str | os.PathLike[str],
        mode: str = "r",
        driver: str | None = None,
        compression: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> BenchmarkRecord:
        """
        Greedy version of ``FileIO.read_batched()``, returning the first read record.
        When reading a multi-record file, this uses as much memory as the batched version.
        """
        records = self.read_batched(
            file=file, mode=mode, driver=driver, compression=compression, options=options
        )
        return records[0]

    def read_batched(
        self,
        file: str | os.PathLike[str],
        mode: str = "r",
        driver: str | None = None,
        compression: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[BenchmarkRecord]:
        """
        Reads a set of benchmark records from the given file path.

        The file driver is chosen based on the extension found on the ``file`` path.

        Parameters
        ----------
        file: str | os.PathLike[str]
            The file name to read from.
        mode: str
            File mode to use. Can be any of the modes used in builtin ``open()``.
        driver: str | None
            File driver implementation to use. If None, the file driver inferred from the
            given file path's extension will be used.
        compression: str | None
            Compression engine to use. If None, the compression inferred from the given
            file path's extension will be used.
        options: dict[str, Any] | None
            Options to pass to the respective file driver implementation.

        Returns
        -------
        list[BenchmarkRecord]
            The benchmark records contained in the file.
        """
        fileext = Path(file).suffix.removeprefix(".")
        # if the extension looks like FORMAT.COMPRESSION, we split.
        if fileext.count(".") == 1:
            # TODO: Are there file extensions with more than one meaningful part?
            ext_driver, ext_compression = fileext.rsplit(".", 1)
        else:
            ext_driver, ext_compression = fileext, None

        driver = driver or ext_driver
        compression = compression or ext_compression

        _, de = get_driver_implementation(driver)

        # canonicalize extension to make sure the file gets it correctly
        # regardless of where driver and compression came from.
        fullext = "." + driver
        if compression is not None:
            fullext += "." + compression
            file = Path(file).with_suffix(fullext)
            fd = get_compression_algorithm(compression)(file, mode)
        else:
            file = Path(file).with_suffix(fullext)
            fd = open(file, mode)

        # dummy value, since the context mode is unused in read ops.
        fdoptions = FileDriverOptions(ctxmode="omit", options=options or {})

        with fd as fp:
            return de(fp, fdoptions)

    def write(
        self,
        record: BenchmarkRecord,
        file: str | os.PathLike[str],
        mode: str = "w",
        driver: str | None = None,
        compression: str | None = None,
        ctxmode: Literal["flatten", "inline", "omit"] = "inline",
        options: dict[str, Any] | None = None,
    ) -> None:
        """Greedy version of ``FileIO.write_batched()``"""
        self.write_batched(
            [record],
            file=file,
            mode=mode,
            driver=driver,
            compression=compression,
            ctxmode=ctxmode,
            options=options,
        )

    def write_batched(
        self,
        records: Sequence[BenchmarkRecord],
        file: str | os.PathLike[str],
        mode: str = "w",
        driver: str | None = None,
        compression: str | None = None,
        ctxmode: Literal["flatten", "inline", "omit"] = "inline",
        options: dict[str, Any] | None = None,
    ) -> None:
        """
        Writes a benchmark record to the given file path.

        The file driver is chosen based on the extension found on the ``file`` path.

        Parameters
        ----------
        records: Sequence[BenchmarkRecord]
            The record to write to the database.
        file: str | os.PathLike[str]
            The file name to write to.
        mode: str
            File mode to use. Can be any of the modes used in builtin ``open()``.
        driver: str | None
            File driver implementation to use. If None, the file driver inferred from the
            given file path's extension will be used.
        compression: str | None
            Compression engine to use. If None, the compression inferred from the given
            file path's extension will be used.
        ctxmode: Literal["flatten", "inline", "omit"]
            How to handle the benchmark context when writing the record data.
        options: dict[str, Any] | None
            Options to pass to the respective file driver implementation.
        """
        fileext = Path(file).suffix.removeprefix(".")
        # if the extension looks like FORMAT.COMPRESSION, we split.
        if fileext.count(".") == 1:
            ext_driver, ext_compression = fileext.rsplit(".", 1)
        else:
            ext_driver, ext_compression = fileext, None

        driver = driver or ext_driver
        compression = compression or ext_compression

        ser, _ = get_driver_implementation(driver)

        # canonicalize extension to make sure the file gets it correctly
        # regardless of where driver and compression came from.
        fullext = "." + driver
        if compression is not None:
            fullext += "." + compression
            file = Path(file).with_suffix(fullext)
            fd = get_compression_algorithm(compression)(file, mode)
        else:
            file = Path(file).with_suffix(fullext)
            fd = open(file, mode)

        fdoptions = FileDriverOptions(ctxmode=ctxmode, options=options or {})
        with fd as fp:
            ser(records, fp, fdoptions)


class FileReporter(FileIO, BenchmarkReporter):
    pass
