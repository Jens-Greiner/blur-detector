# Blur Detector

Scan images (or folders of images) and score each one for blurriness using the
**variance of the Laplacian** — a low score means few sharp edges, i.e. a blurry
image. Below a threshold you choose, images can be moved to a folder or deleted.

There are two front-ends over a shared core (`blur_core.py`): a PyQt6 desktop
app (`main_gui.py`) and an interactive terminal app (`main.py`).

## Download (Windows)

Grab the latest **`BlurDetector.exe`** from the
[Releases page](../../releases/latest). No Python install required — download,
double-click, and go.

> **Note:** the executable is unsigned, so Windows SmartScreen or your antivirus
> may warn on first launch. Choose *More info → Run anyway*. The build is
> produced automatically on a GitHub Actions Windows runner (see
> `.github/workflows/build-windows.yml`).

## Run from source (any platform)

Uses [uv](https://docs.astral.sh/uv/) and requires Python 3.13.

```bash
uv run main_gui.py             # PyQt6 GUI
uv run main.py                 # CLI, prompts for input
uv run main.py /path/to/images # CLI with a path argument
```

## Supported formats

`.jpg .jpeg .png .gif .bmp .tiff .webp` plus RAW `.cr3 .dng .nef .arw`.
