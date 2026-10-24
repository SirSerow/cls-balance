import numpy as np
import pytest
import torch
from PIL import Image

from cls_balance.transforms import Preprocessing


@pytest.mark.parametrize("size", [(400, 120), (100, 420), (224, 244), (1, 500)])
def test_banner_and_entire_content_preserved(size):
    image = Image.new("RGB", size, "white")
    image.paste((255, 0, 0), (0, size[1] - 20, size[0], size[1]))
    processed = np.array(Preprocessing().image(image))
    assert processed.shape == (224, 224, 3)
    # No banner pixels survive, and the long side touches both square edges.
    assert np.array_equal(processed[:, :, 0], processed[:, :, 1])
    if size[1] - 20 >= size[0]:
        assert np.any(processed[0] == 255) and np.any(processed[-1] == 255)
    else:
        assert np.any(processed[:, 0] == 255) and np.any(processed[:, -1] == 255)


def test_boundary_markers_retained():
    image = Image.new("RGB", (224, 132), "white")
    image.paste((0, 255, 0), (0, 0, 5, 112))
    image.paste((0, 0, 255), (219, 0, 224, 112))
    processed = np.array(Preprocessing().image(image))
    assert (processed[56:168, :5] == (0, 255, 0)).all()
    assert (processed[56:168, -5:] == (0, 0, 255)).all()


def test_weights_normalization_and_eval_determinism():
    preprocessing = Preprocessing()
    image = Image.new("RGB", (224, 244), "white")
    tensor = preprocessing.tensor(image)
    expected = (torch.ones(3) - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(tensor[:, 100, 100], expected)
    assert torch.equal(tensor, preprocessing.tensor(image, seed=1, epoch=99))


def test_augmentation_is_per_image_seed_epoch():
    image = Image.fromarray(np.arange(80 * 80 * 3, dtype=np.uint8).reshape(80, 80, 3))
    p = Preprocessing()
    a = p.tensor(image, training=True, seed=42, epoch=1, image_id="0000001")
    assert torch.equal(a, p.tensor(image, training=True, seed=42, epoch=1, image_id="0000001"))
    assert not torch.equal(a, p.tensor(image, training=True, seed=42, epoch=2, image_id="0000001"))
    assert not torch.equal(a, p.tensor(image, training=True, seed=43, epoch=1, image_id="0000001"))


@pytest.mark.parametrize("kwargs", [{"image_size": 256}, {"remove_bottom_banner_px": 0},
                                  {"pad_rgb": (256, 0, 0)}, {"horizontal_flip_probability": -1},
                                  {"contrast_jitter": 2}, {"interpolation": "nearest"}])
def test_invalid_protocol(kwargs):
    with pytest.raises(ValueError):
        Preprocessing(**kwargs)


def test_preprocessing_hash_changes():
    assert Preprocessing().sha256 != Preprocessing(pad_rgb=(10, 10, 10)).sha256
