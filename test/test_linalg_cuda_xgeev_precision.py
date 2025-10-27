import torch
import numpy as np

import unittest
import itertools
import warnings
import math
from math import inf, nan, isnan
import re
import random
from random import randrange
from itertools import product
from functools import reduce, partial
from typing import Union, Optional
from torch._prims_common import DimsType
from packaging import version

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, TEST_SCIPY, IS_MACOS, IS_WINDOWS, slowTest,
     TEST_WITH_ROCM, IS_FBCODE, IS_REMOTE_GPU, iter_indices,
     make_fullrank_matrices_with_distinct_singular_values,
     freeze_rng_state, IS_ARM64, IS_SANDCASTLE, TEST_OPT_EINSUM, parametrize, skipIfTorchDynamo,
     setBlasBackendsToDefaultFinally, setLinalgBackendsToDefaultFinally, serialTest,
     runOnRocmArch, MI300_ARCH, TEST_CUDA)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, has_cusolver, has_hipsolver,
     onlyCPU, skipCUDAIfNoMagma, skipCPUIfNoLapack, precisionOverride,
     skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, onlyNativeDeviceTypes, dtypesIfCUDA,
     onlyCUDA, skipMeta, skipCUDAIfNoCusolver, skipCUDAIfNotRocm, skipCUDAIfRocmVersionLessThan,
     dtypesIfMPS, largeTensorTest)
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
    all_types, all_types_and_complex_and, floating_and_complex_types, integral_types,
    floating_and_complex_types_and, floating_types_and, complex_types,
)
from torch.testing._internal.common_cuda import SM53OrLater, SM80OrLater, SM90OrLater, tf32_on_and_off, _get_magma_version, \
    _get_torch_cuda_version, CDNA2OrLater, TEST_MULTIGPU
from torch.testing._internal.common_quantization import _group_quantize_tensor, _dynamically_quantize_per_channel, \
    _group_quantize_tensor_symmetric
from torch.testing._internal.common_mkldnn import reduced_f32_on_and_off
from torch.distributions.binomial import Binomial
import torch.backends.opt_einsum as opt_einsum
import operator
import contextlib

# Protects against includes accidentally setting the default dtype
assert torch.get_default_dtype() is torch.float32

if TEST_SCIPY:
    import scipy


@onlyCUDA
@skipCUDAIfNoMagma
@dtypes(*floating_and_complex_types())
def test_eig_compare_backends_eigen_equation(self, device, dtype):
    def run_test(shape, *, symmetric=False):
        from torch.testing._internal.common_utils import random_symmetric_matrix

        if not dtype.is_complex and symmetric:
            # for symmetric real-valued inputs eigenvalues and eigenvectors have imaginary part equal to zero
            a = random_symmetric_matrix(shape[-1], *shape[:-2], dtype=dtype, device=device)
        else:
            a = make_tensor(shape, dtype=dtype, device=device)

        actual = torch.linalg.eig(a)

        complementary_device = 'cpu'
        a_complementary_device = a.to(complementary_device)
        # compare with CPU
        expected = torch.linalg.eig(a_complementary_device)
        self.assertEqual(expected[0], actual[0]) #as eigenvalues should be identical, we check them against each other
        self.assertEqual(expected[1].numel, actual[1].numel) #eigenvectors can differ by a complex factor of abs 1, so we don't check them directly, but check their number

        # check precision using eigendecomposition identity
        diff_expected = (a_complementary_device@expected[1]) - (expected[1] * expected[0].unsqueeze(-2))
        diff_actual = (a@actual[1]) - (actual[1] * actual[0].unsqueeze(-2))

        if dtype in [torch.float32, torch.complex64]:
            atol = 1e-1
        else:
            atol = 1e-13

        passing = False

        #check if we are either down to the expected precision from my comparisons of NumPy and CuSolver
        # or if we at least surpassed the cpu path satisfying the eigendecomposition identity than expected
        if diff_actual <= atol or diff_actual <= diff_expected:
            passing = True

        self.assertTrue(passing)


    shapes = [(0, 0),  # Empty matrix
              (5, 5),  # Single matrix
              (0, 0, 0), (0, 5, 5),  # Zero batch dimension tensors
              (2, 5, 5),  # 3-dim tensors
              (2, 1, 5, 5)]  # 4-dim tensors
    for shape in shapes:
        run_test(shape)
        run_test(shape, symmetric=True)


