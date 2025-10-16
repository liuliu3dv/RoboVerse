import numpy as np
from PIL import Image
from typing import Union

from pyquaternion import Quaternion

def quaternion_multiply(
    init_quat: list[float], rotate_quat: list[float]
) -> list[float]:
    qx, qy, qz, qw = init_quat
    q1 = Quaternion(w=qw, x=qx, y=qy, z=qz)
    qx, qy, qz, qw = rotate_quat
    q2 = Quaternion(w=qw, x=qx, y=qy, z=qz)
    quat = q2 * q1

    return [quat.x, quat.y, quat.z, quat.w]


def alpha_blend_rgba(
    fg_image: Union[str, Image.Image, np.ndarray],
    bg_image: Union[str, Image.Image, np.ndarray],
) -> Image.Image:
    """Alpha blends a foreground RGBA image over a background RGBA image.

    Args:
        fg_image: Foreground image. Can be a file path (str), a PIL Image,
            or a NumPy ndarray.
        bg_image: Background image. Can be a file path (str), a PIL Image,
            or a NumPy ndarray.

    Returns:
        A PIL Image representing the alpha-blended result in RGBA mode.
    """
    if isinstance(fg_image, str):
        fg_image = Image.open(fg_image)
    elif isinstance(fg_image, np.ndarray):
        fg_image = Image.fromarray(fg_image)

    if isinstance(bg_image, str):
        bg_image = Image.open(bg_image)
    elif isinstance(bg_image, np.ndarray):
        bg_image = Image.fromarray(bg_image)

    if fg_image.size != bg_image.size:
        raise ValueError(
            f"Image sizes not match {fg_image.size} v.s. {bg_image.size}."
        )

    fg = fg_image.convert("RGBA")
    bg = bg_image.convert("RGBA")

    return Image.alpha_composite(bg, fg)