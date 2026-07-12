"""Shared blur-detection core used by both the CLI (main.py) and GUI (main_gui.py).

Blur is measured by the variance of the Laplacian: low variance means few sharp
edges, i.e. a blurry image. Higher score = sharper.

Before scoring, every image is downscaled so its longest edge is at most
``NORMALIZE_LONGEST_EDGE`` px. Laplacian variance is resolution-dependent, so
without this a 24MP file and a downsized 2MP copy of the same scene would score
very differently and a single threshold would be meaningless. Normalizing makes
scores comparable across resolutions and across RAW vs non-RAW inputs.
"""

import hashlib
import json
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import rawpy
import cv2

# Supported image extensions. RAW formats are decoded via rawpy; the rest via
# cv2.imread. (SVG is intentionally excluded: cv2.imread cannot decode it.)
RAW_EXTENSIONS: List[str] = ['.cr3', '.dng', '.nef', '.arw']
IMAGE_EXTENSIONS: List[str] = [
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',
] + RAW_EXTENSIONS

# Longest-edge (px) that images are downscaled to before scoring. Makes the blur
# score resolution-invariant so one threshold works across a mixed set. Only
# downscaling is applied — images already smaller than this are left untouched so
# their scores are never inflated.
NORMALIZE_LONGEST_EDGE: int = 1000

# Bumped whenever the scoring algorithm changes so stale cache entries are
# ignored. The introduction of NORMALIZE_LONGEST_EDGE is version 2.
CACHE_VERSION: int = 2


def _decode_image(image_path: str) -> Optional["np.ndarray"]:
    """Decode an image (RAW or standard) to a BGR ``np.ndarray``, or ``None``.

    RAW files go through ``rawpy`` (fast half-size LINEAR demosaic); everything
    else through ``cv2.imread``. Returns ``None`` on any failure so callers can
    treat unreadable files uniformly.
    """
    file_extension = os.path.splitext(image_path)[1].lower()

    if file_extension in RAW_EXTENSIONS:
        with rawpy.imread(image_path) as raw:
            # Fast demosaicing for better performance.
            rgb_image = raw.postprocess(rawpy.Params(
                demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                use_camera_wb=True,
                no_auto_bright=True,
                half_size=True,
            ))
        # rawpy returns RGB; convert to BGR to match cv2.imread's convention.
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    return cv2.imread(image_path)


def _normalize_for_scoring(gray_image: "np.ndarray") -> "np.ndarray":
    """Downscale a grayscale image so its longest edge is NORMALIZE_LONGEST_EDGE.

    Only ever downscales (never upscales). Uses ``INTER_AREA``, the recommended
    interpolation for shrinking.
    """
    height, width = gray_image.shape[:2]
    longest = max(height, width)
    if longest <= NORMALIZE_LONGEST_EDGE:
        return gray_image

    scale = NORMALIZE_LONGEST_EDGE / longest
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(gray_image, new_size, interpolation=cv2.INTER_AREA)


def calculate_blurriness(image_path: str) -> Tuple[str, Optional[float]]:
    """Process a single image and calculate its blur score.

    Returns a tuple ``(image_path, variance)`` where ``variance`` is ``None`` on
    any failure (missing/unreadable/corrupt file). The image is normalized to a
    fixed longest edge before scoring (see module docstring). Designed to run in
    a separate process, so it must stay at module level to remain picklable.
    """
    try:
        if not os.path.exists(image_path):
            return (image_path, None)

        bgr_image = _decode_image(image_path)
        if bgr_image is None:
            return (image_path, None)

        # Grayscale, normalize resolution, then variance of the Laplacian.
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        gray_image = _normalize_for_scoring(gray_image)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        variance = laplacian.var()

        return (image_path, float(variance))

    except rawpy.FileOpenError:
        return (image_path, None)
    except Exception:
        return (image_path, None)


def get_image_paths(path: str) -> List[str]:
    """Find all supported image files within a path (a single file or directory).

    Returns a list of absolute paths to image files. Non-existent paths and
    unsupported file types yield an empty list.
    """
    image_files: List[str] = []

    normalized_path = os.path.abspath(path)

    if not os.path.exists(normalized_path):
        return image_files

    # Case 1: a single file.
    if os.path.isfile(normalized_path):
        _, file_extension = os.path.splitext(normalized_path)
        if file_extension.lower() in IMAGE_EXTENSIONS:
            image_files.append(normalized_path)
        return image_files

    # Case 2: a directory — walk it recursively.
    if os.path.isdir(normalized_path):
        for dirpath, _, filenames in os.walk(normalized_path):
            for filename in filenames:
                _, file_extension = os.path.splitext(filename)
                if file_extension.lower() in IMAGE_EXTENSIONS:
                    image_files.append(os.path.join(dirpath, filename))

    return image_files


def unique_destination(dest_dir: str, filename: str) -> str:
    """Return a path in ``dest_dir`` for ``filename`` that does not overwrite.

    If ``filename`` already exists in the directory, a numeric suffix is inserted
    before the extension: ``photo.jpg`` -> ``photo (1).jpg`` -> ``photo (2).jpg``,
    and so on until a free name is found.
    """
    candidate = os.path.join(dest_dir, filename)
    if not os.path.exists(candidate):
        return candidate

    stem, ext = os.path.splitext(filename)
    counter = 1
    while True:
        candidate = os.path.join(dest_dir, f"{stem} ({counter}){ext}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


# ---------------------------------------------------------------------------
# Score cache
#
# Re-running analysis on a folder is common (e.g. to try a new threshold), and
# decoding — RAW especially — is expensive. The cache stores scores keyed by
# (mtime, size) so unchanged files are skipped on a re-run. It lives in the OS
# cache directory rather than inside scanned folders, keyed by the set of roots
# being analyzed, and is invalidated automatically when CACHE_VERSION changes.
# ---------------------------------------------------------------------------

def _cache_dir() -> str:
    """Directory where cache files live (created if needed)."""
    if os.name == "nt":
        # Windows: prefer %LOCALAPPDATA% so the cache lands in the conventional
        # per-user cache location rather than the home-directory root.
        base = os.environ.get("LOCALAPPDATA") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
    path = os.path.join(base, "blur-detector")
    os.makedirs(path, exist_ok=True)
    return path


def cache_path_for(roots: List[str]) -> str:
    """Return the cache file path for a given set of analysis roots.

    The filename is a stable hash of the absolute, sorted roots so the same
    selection maps to the same cache file across runs.
    """
    key = "\n".join(sorted(os.path.abspath(r) for r in roots))
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(_cache_dir(), f"{digest}.json")


def load_cache(cache_file: str) -> Dict[str, dict]:
    """Load a score cache, returning ``{path: {mtime, size, score}}``.

    Returns an empty dict if the file is missing, unreadable, or was written by a
    different ``CACHE_VERSION``.
    """
    try:
        with open(cache_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("version") != CACHE_VERSION:
            return {}
        entries = data.get("entries", {})
        return entries if isinstance(entries, dict) else {}
    except (OSError, ValueError):
        return {}


def save_cache(cache_file: str, entries: Dict[str, dict]) -> None:
    """Write ``entries`` to ``cache_file`` (best-effort; ignores write errors).

    Writes atomically via a temp file + replace so a crash mid-write can't leave
    a corrupt cache.
    """
    payload = {"version": CACHE_VERSION, "entries": entries}
    try:
        directory = os.path.dirname(cache_file) or "."
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, cache_file)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    except OSError:
        pass


def _file_signature(path: str) -> Optional[Tuple[float, int]]:
    """Return ``(mtime, size)`` for ``path``, or ``None`` if it can't be stat'd."""
    try:
        st = os.stat(path)
        return (st.st_mtime, st.st_size)
    except OSError:
        return None


def cached_score(entries: Dict[str, dict], path: str) -> Optional[float]:
    """Return a cached score for ``path`` if present and still valid.

    Validity is decided by comparing the file's current ``(mtime, size)`` against
    the cached signature. Returns ``None`` on a miss or a stale entry. Note a
    valid cached score may itself be ``None`` (a file that failed to decode last
    time); use :func:`has_valid_cache_entry` to distinguish miss from cached-None.
    """
    entry = entries.get(path)
    if not entry:
        return None
    sig = _file_signature(path)
    if sig is None or entry.get("mtime") != sig[0] or entry.get("size") != sig[1]:
        return None
    return entry.get("score")


def has_valid_cache_entry(entries: Dict[str, dict], path: str) -> bool:
    """True if ``entries`` holds a still-valid entry for ``path`` (score may be None)."""
    entry = entries.get(path)
    if not entry:
        return False
    sig = _file_signature(path)
    return sig is not None and entry.get("mtime") == sig[0] and entry.get("size") == sig[1]


def make_cache_entry(path: str, score: Optional[float]) -> Optional[dict]:
    """Build a cache entry ``{mtime, size, score}`` for ``path``, or ``None``.

    Returns ``None`` if the file can't be stat'd (so it simply isn't cached).
    """
    sig = _file_signature(path)
    if sig is None:
        return None
    return {"mtime": sig[0], "size": sig[1], "score": score}
