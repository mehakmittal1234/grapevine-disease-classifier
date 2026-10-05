"""Model factory: ImageNet-pretrained backbones with a new 4-class head.

Fine-tuning policy (identical for all three architectures):

* phase ``head``     - whole backbone frozen, only the new classifier head trains;
* phase ``finetune`` - the early, generic stages (stem up to the stride-8 / 28x28 stage) stay
  frozen and the deeper stages plus the head train.

Freezing the early stages roughly halves step time and GPU memory on an 8 GB Apple M2
(measured: ResNet50 9.7 -> 21.0 img/s, EfficientNet-B0 13.9 -> 29.3 img/s) at little cost,
because those layers learn generic edges and textures. Frozen modules are kept in eval mode
so their BatchNorm statistics stay at the ImageNet values.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tvm

from .config import MODEL_NAMES


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "mobilenet_v2":
        model = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "resnet50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        # Same dropout rate as the classifiers of the other two torchvision models.
        model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, num_classes))
    elif name == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    return model


def head_module(model: nn.Module, name: str) -> nn.Module:
    return model.fc if name == "resnet50" else model.classifier


def early_stages(model: nn.Module, name: str) -> list[nn.Module]:
    """Stem and stages up to stride 8 (28x28 feature maps for a 224 input)."""
    if name == "mobilenet_v2":
        return list(model.features[:7])  # stem + 16/24/32-channel inverted-residual blocks
    if name == "resnet50":
        return [model.conv1, model.bn1, model.layer1, model.layer2]
    if name == "efficientnet_b0":
        return list(model.features[:4])  # stem + 16/24/40-channel MBConv stages
    raise ValueError(name)


def gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Last convolutional block (7x7 maps), the usual Grad-CAM target."""
    if name == "resnet50":
        return model.layer4[-1]
    return model.features[-1]  # final 1x1 conv block of MobileNetV2 / EfficientNet-B0


def configure_phase(model: nn.Module, name: str, phase: str) -> list[nn.Module]:
    """Set ``requires_grad`` for a training phase; return the modules that must stay in eval mode."""
    head = head_module(model, name)
    if phase == "head":
        frozen = [m for m in model.children() if m is not head]
    elif phase == "finetune":
        frozen = early_stages(model, name)
    else:
        raise ValueError(phase)
    for p in model.parameters():
        p.requires_grad_(True)
    for module in frozen:
        for p in module.parameters():
            p.requires_grad_(False)
    return frozen


def set_train_mode(model: nn.Module, frozen: list[nn.Module]) -> None:
    model.train()
    for module in frozen:
        module.eval()


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def load_checkpoint(path, device: torch.device):
    """Load one of our checkpoints and rebuild its model in eval mode on ``device``.

    ``weights_only=True`` refuses arbitrary pickled objects; the only extra type the file
    needs is ``TorchVersion`` (the str subclass stored in its environment metadata).
    """
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["model_name"], len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval().to(device), ckpt
