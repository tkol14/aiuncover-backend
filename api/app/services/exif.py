
from PIL import Image
from io import BytesIO

def has_exif(image_bytes: bytes) -> bool:
    try:
        img = Image.open(BytesIO(image_bytes))
        exif = img.getexif()
        return exif is not None and len(exif.items()) > 0
    except Exception:
        return False
