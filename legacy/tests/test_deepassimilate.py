import torch
from deepassimilate import SRCNN, generate_random_image

def test_srcnn():
    model = SRCNN()
    input_tensor = torch.randn(1, 1, 32, 32)  # Batch size of 1, 1 channel, 32x32 image
    output_tensor = model(input_tensor)
    assert output_tensor.shape == (1, 1, 32, 32)

def test_random_image():
    image = generate_random_image(42, (256, 256))
    assert image.shape == (256, 256)
