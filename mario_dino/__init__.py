"""MARIO Tasks 1-2 with DINOv2, ConvNeXt or VisionFM, a GRU forecaster and a shared fusion head.

All encoders support frozen feature extraction. ConvNeXt/VisionFM additionally support
full image-based Task 1 fine-tuning; DINOv2 retains its token-based partial trainer.

Importing this package must never import TensorFlow; OCTIP is only imported lazily by
``mario_dino.octip_cache`` in ``--mode octip``.
"""

__version__ = "0.1.0"
