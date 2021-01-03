from ..preprocessing import std, ProperCenterAndScale
import numpy as np
import tensorly as tl
from ...testing import assert_
import pytest


@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("ddof", [None, 0, 1])
@pytest.mark.parametrize("axis", [None, 0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)])
def test_std(axis, keepdims, ddof):
    np.random.seed(1234)
    x = np.random.standard_normal((5, 6, 7))
    tensor = tl.tensor(x)

    if ddof is not None:
        tl_std = std(tensor, axis=axis, keepdims=keepdims, ddof=ddof)
        np_std = np.std(x, axis=axis, keepdims=keepdims, ddof=ddof)
    else:
        tl_std = std(tensor, axis=axis, keepdims=keepdims,)
        np_std = np.std(x, axis=axis, keepdims=keepdims,)
    
    np.testing.assert_allclose(tl.to_numpy(tl_std), np_std)

        