import math

import numpy as np
import pytest

import nnbench

N = 1024
NUM_REPLICAS = 8
SAFETY_FACTOR = 1.1

NP_MATSIZE_BYTES = N**2 * np.float64().itemsize + 128
# for a matmul, we alloc LHS, RHS, and RESULT.
# math.ceil gives us a cushion of up to 1MB, which is for other system allocs.
EXPECTED_MEM_USAGE_MB = math.ceil(3 * NP_MATSIZE_BYTES / 1_000_000)


@pytest.mark.limit_memory(f"{EXPECTED_MEM_USAGE_MB}MB")
def test_foobar():
    """Checks that a benchmark family works with GC."""

    @nnbench.parametrize({"b": np.random.randn(N, N)} for _ in range(NUM_REPLICAS))
    def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b

    a = np.random.randn(N, N)
    nnbench.run(matmul, params={"a": a})
