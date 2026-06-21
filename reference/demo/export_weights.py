"""Export trained GPT weights to a compact ``.npz`` for the browser demo.

The numpy demo only needs the parameter arrays. Casting to float16 and using
compressed storage keeps the download small enough to ship to a web page.
"""

import numpy as np

from reference.demo.numpy_forward import weights_from_torch


def export(model, path):
    """Save every parameter of ``model`` to ``path`` as fp16, compressed.

    Loading back with ``np.load`` and casting to float32 reproduces weights
    suitable for ``numpy_forward`` (fp16 storage trades a little precision for
    a much smaller file).
    """
    weights = weights_from_torch(model)
    fp16 = {name: arr.astype(np.float16) for name, arr in weights.items()}
    np.savez_compressed(path, **fp16)
    return path
