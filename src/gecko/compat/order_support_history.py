"""Read inventoried order-support research snapshots without an old namespace."""
from __future__ import annotations

import codecs
import collections
import pickle
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gecko.research.order_support import RNGSnapshot


class _OrderSupportUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        multiarray = import_module(
            "numpy._core.multiarray" if int(np.__version__.split(".")[0]) >= 2
            else "numpy.core.multiarray"
        )
        allowed = {
            ("begin.uefa.experiments.order_support", "RNGSnapshot"): RNGSnapshot,
            ("gecko.research.order_support", "RNGSnapshot"): RNGSnapshot,
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("_codecs", "encode"): codecs.encode,
            ("numpy", "dtype"): np.dtype,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy.core.multiarray", "_reconstruct"): multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): multiarray._reconstruct,
            ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
            **{("torch", kind + "Storage"): getattr(torch, kind + "Storage")
               for kind in ("Bool", "Byte", "Float", "Long")},
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported order-support snapshot global: {module}.{name}")
        return allowed[(module, name)]


class _OrderSupportPickle:
    __name__ = "gecko.compat.order_support_history"
    Unpickler = _OrderSupportUnpickler


def load_order_support_snapshot(path: str | Path) -> Any:
    """Load the inventoried RNG/tensor format through an exact global allowlist."""
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=_OrderSupportPickle)
