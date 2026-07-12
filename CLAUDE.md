# CLAUDE.md

Guidance for working in this repository.

## What this is

Blur Detector scans images (or folders of images) and computes a blur score for
each using the **variance of the Laplacian** — low variance means few sharp
edges, i.e. a blurry image. Below a user-chosen threshold, images can be moved
to a folder or deleted.

There are two independent front-ends over a shared core:

- **`blur_core.py`** — the shared scoring logic. `calculate_blurriness(path)`
  returns `(path, score|None)`; `get_image_paths(path)` walks a file or
  directory for supported images. Both front-ends import from here, so algorithm
  / format / RAW changes are made in **one** place.
- **`main.py`** — interactive terminal app. Prompts for paths (CLI arg, GUI file
  dialog via tkinter, or manual entry), prints a text progress bar and an ASCII
  histogram of the score distribution, then asks for a threshold and a
  move/delete action.
- **`main_gui.py`** — PyQt6 desktop app. A 3-step wizard (Select → Analyze →
  Filter) with a custom clickable histogram widget, background thumbnail
  generation, and move/delete actions.
- **`styles.py`** — Material Design 3 color palette (`Colors`) and the Qt
  stylesheet (`get_stylesheet()`) consumed only by `main_gui.py`.

`main_gui.py` keeps a thin module-level `process_image_worker` wrapper around
`calculate_blurriness` as the picklable target for its `multiprocessing.Pool`.

## Running

Uses [uv](https://docs.astral.sh/uv/) (see `pyproject.toml`, requires Python
3.13). RAW support (`rawpy`), OpenCV, matplotlib, and PyQt6 are dependencies.

```bash
uv run main.py                 # CLI, prompts for input
uv run main.py /path/to/images # CLI with a path argument
uv run main_gui.py             # PyQt6 GUI
```

## Key details

- **Algorithm:** grayscale → `cv2.Laplacian(..., CV_64F)` → `.var()`. Higher
  score = sharper.
- **Supported formats:** `.jpg .jpeg .png .gif .bmp .tiff .webp` plus RAW
  `.cr3 .dng .nef .arw` (see `IMAGE_EXTENSIONS` in `blur_core.py`). RAW files go
  through `rawpy.postprocess` with `half_size=True` and LINEAR demosaic for
  speed; everything else uses `cv2.imread`.
- **Parallelism:** both front-ends use `multiprocessing.Pool` with
  `cpu_count()` processes and `imap_unordered` for streaming progress. Worker
  functions are module-level (required for pickling) and return
  `(path, score)`, with `score = None` on any failure. `main.py` calls
  `freeze_support()` for frozen builds. In the GUI the pool runs inside an
  `AnalysisThread` (`QThread`) so the UI stays responsive.
- **Failures are silent by design:** unreadable/corrupt files yield
  `score = None` and are excluded from results, histograms, and filtering rather
  than raising.

## GUI structure (`main_gui.py`)

- `BlurDetectorGUI` — `QMainWindow` with a `QStackedWidget` of 3 steps.
- `HistogramWidget` — custom `QWidget` that paints the score distribution in
  fixed bins of width 5.0; clicking a bar emits `bin_clicked` and lists the
  images in that range.
- `ThumbnailWorker` (`QThread`) — generates `QImage` thumbnails off the UI
  thread; workers are held in `self._thumb_workers` to avoid GC.
- "Open Selected" / double-click reveals the file in the OS file manager
  (Finder / Explorer / xdg-open), it does not open an in-app preview.

## Conventions

- Type hints throughout; functions carry docstrings — match that style.
- Destructive actions (delete) require explicit confirmation: the CLI demands
  typing `YES`; the GUI shows a Yes/No dialog. Preserve this guard.
- macOS: `QT_MAC_WANTS_LAYER=1` is set before importing PyQt6 — keep that import
  ordering.
- `README.md` is currently empty.
