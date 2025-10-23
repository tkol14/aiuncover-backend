
import hashlib
import random

def score_model_stub(image_bytes: bytes) -> float:
    """Deterministic-ish stub model score in [0,1] based on file hash."""
    h = hashlib.sha256(image_bytes).hexdigest()
    seed = int(h[:8], 16)
    rng = random.Random(seed)
    return round(rng.random(), 4)
