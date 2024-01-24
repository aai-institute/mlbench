import os

import pytest

from nnbench import runner


def test_argcheck(typecheckfolder: str) -> None:
    benchmarks = os.path.join(typecheckfolder, "benchmarks.py")
    r = runner.AbstractBenchmarkRunner()
    with pytest.raises(TypeError, match="expected type <class 'int'>.*"):
        r.run(benchmarks, params={"x": 1, "y": "1"})
    with pytest.raises(ValueError, match="missing value for required parameter.*"):
        r.run(benchmarks, params={"x": 1})

    r.run(benchmarks, params={"x": 1, "y": 1})


def test_error_on_duplicate_params(typecheckfolder: str) -> None:
    benchmarks = os.path.join(typecheckfolder, "duplicate_benchmarks.py")

    with pytest.raises(TypeError, match="got non-unique types.*"):
        r = runner.AbstractBenchmarkRunner()
        r.run(benchmarks, params={"x": 1, "y": 1})