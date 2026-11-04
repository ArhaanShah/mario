"""Native timm ConvNeXt pre-classifier embeddings; no runtime downloads."""
from torch import nn


def build_convnext(name, pretrained=False, **kwargs):
    import timm

    if name != "convnext_tiny.in12k_ft_in1k":
        raise ValueError(f"unsupported ConvNeXt name: {name!r}")
    return timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg", **kwargs)


class ConvNeXtEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.out_dim = int(model.num_features)

    def forward(self, images):
        return self.model(images)
