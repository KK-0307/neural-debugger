"""A compact transformer encoder implemented from scratch on PyTorch tensors.

Only ``nn.Linear``, ``nn.Embedding`` and autograd are used as primitives; scaled dot-product
attention, multi-head splitting, masking, layer normalisation, residual blocks, pooling,
projection and L2 normalisation are written out explicitly (see ``docs/ml-model.md``).
"""

from .config import TransformerConfig
from .model import DualEncoder, SequenceEncoder, TransformerEncoder

__all__ = ["DualEncoder", "SequenceEncoder", "TransformerConfig", "TransformerEncoder"]
