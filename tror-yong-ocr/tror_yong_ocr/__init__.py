from collections import namedtuple
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms
from .model import TrorYongOCRModel, TrorYongOCRConfig
from .tokenizer import get_tokenizer, CharTokenizer


Result = namedtuple("Result", "output_ids text")


def patchify(img, patch_size=(4, 8)):
    """
    Args:
        img: Tensor of shape (C, H, W)
        patch_size: int, size of the square patch (P0, P1)
    Returns:
        patches: Tensor of shape (L0, patch_dim) where patch_dim = C * P0 * P1
    """

    # 1. Use unfold to extract patches
    # unfold(dimension, size, step)
    # This extracts sliding blocks of size patch_size
    patches = F.unfold(img.unsqueeze(
        0), kernel_size=patch_size, stride=patch_size)

    # After unfold, shape is (1, C*P0*P1, L0)
    # 2. Reshape and permute to (L0, patch_dim)
    patches = patches.squeeze(0).transpose(0, 1)

    return patches


def recognize(image: Image, model: TrorYongOCRModel, tokenizer: CharTokenizer, max_tokens: int, temperature=1.0, top_k=None, seed=168) -> Result:
    preprocess = transforms.Compose(
        [
            transforms.Resize((32, )),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ])
    img_tensor = preprocess(image)
    patches = patchify(img_tensor, model.config.patch_size)
    assert patches.shape[0] <= model.config.block_size, f"Only support image aspect ratio smaller than 32, your image: width={image.width}, height={image.height}"
    patches = patches.to(model.device)
    output_ids = model.decode(patches, max_tokens, temperature, top_k, seed)
    text = tokenizer.decode(output_ids[0].tolist(), ignore_special_tokens=True)
    return Result(output_ids=output_ids[0], text=text)
