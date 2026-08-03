from __future__ import annotations

import torch
import torch.nn.functional as F

from vit_reimplementation import PatchEmbedding, ViT


def test_patchify_preserves_expected_patch_order() -> None:
    embedding = PatchEmbedding(
        image_size=(4, 4),
        patch_size=(2, 2),
        channels=1,
        dim=2,
    )

    image = torch.tensor(
        [[[[
            0.0, 1.0, 2.0, 3.0,
        ], [
            4.0, 5.0, 6.0, 7.0,
        ], [
            8.0, 9.0, 10.0, 11.0,
        ], [
            12.0, 13.0, 14.0, 15.0,
        ]]]]
    )

    expected = torch.tensor(
        [[
            [0.0, 1.0, 4.0, 5.0],
            [2.0, 3.0, 6.0, 7.0],
            [8.0, 9.0, 12.0, 13.0],
            [10.0, 11.0, 14.0, 15.0],
        ]]
    )

    assert torch.equal(
        embedding.patchify(image),
        expected,
    )


def test_vit_forward_and_backward_contract() -> None:
    torch.manual_seed(0)

    batch_size = 2
    image_size = (32, 32)
    patch_size = (4, 4)
    channels = 3
    dim = 96
    num_layers = 3
    num_heads = 4
    hidden_dim = 384
    num_classes = 10

    num_patches = (
        (image_size[0] // patch_size[0])
        * (image_size[1] // patch_size[1])
    )
    sequence_length = num_patches + 1

    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        channels=channels,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )

    images = torch.randn(
        batch_size,
        channels,
        image_size[0],
        image_size[1],
    )

    logits, all_weights = model(images)

    assert logits.shape == (batch_size, num_classes)
    assert logits.is_floating_point()
    assert torch.isfinite(logits).all()
    assert len(all_weights) == num_layers

    for weights in all_weights:
        assert weights.shape == (
            batch_size,
            num_heads,
            sequence_length,
            sequence_length,
        )

        expected_sums = torch.ones(
            batch_size,
            num_heads,
            sequence_length,
            device=weights.device,
        )

        assert torch.allclose(
            weights.sum(dim=-1),
            expected_sums,
            atol=1e-6,
        )
        assert torch.isfinite(weights).all()

    labels = torch.randint(
        low=0,
        high=num_classes,
        size=(batch_size,),
    )

    loss = F.cross_entropy(logits, labels)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    important_parameters = {
        "patch projection": model.patch_embedding.project.weight,
        "CLS token": model.cls_token,
        "query projection": (
            model.transformer.layers[0]
            .attention.project_query.weight
        ),
        "classification head": model.project_classes.weight,
    }

    for name, parameter in important_parameters.items():
        assert parameter.grad is not None, f"No gradient reached {name}"
        assert torch.isfinite(parameter.grad).all(), (
            f"Non-finite gradient found in {name}"
        )
