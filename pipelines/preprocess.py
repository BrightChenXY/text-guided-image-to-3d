from PIL import Image, ImageOps


try:
    _RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    _RESAMPLE = Image.LANCZOS


def preprocess_image(image: Image.Image, size: int = 512) -> Image.Image:
    if image is None:
        raise ValueError("Input image is required for preprocessing.")

    if size <= 0:
        raise ValueError("Preprocess size must be greater than 0.")

    normalized = ImageOps.exif_transpose(image).convert("RGB")
    fitted = ImageOps.contain(normalized, (size, size), _RESAMPLE)

    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    offset_x = (size - fitted.width) // 2
    offset_y = (size - fitted.height) // 2
    canvas.paste(fitted, (offset_x, offset_y))
    return canvas
