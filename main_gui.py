import sys
import os
import shutil
import traceback
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count, freeze_support

# On macOS Qt6 needs this environment variable set before importing PyQt
if sys.platform == 'darwin':
    os.environ.setdefault('QT_MAC_WANTS_LAYER', '1')

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog, QMessageBox,
    QProgressBar, QComboBox, QSpinBox, QFrame, QScrollArea, QListWidget,
    QListWidgetItem, QStackedWidget, QDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF, QPointF
from PyQt6.QtGui import QColor, QIcon, QPixmap, QPainter, QPen, QBrush, QImage, QFont
from send2trash import send2trash
import cv2
import numpy as np
from blur_core import (
    calculate_blurriness,
    get_image_paths,
    unique_destination,
    _decode_image,
    RAW_EXTENSIONS,
    cache_path_for,
    load_cache,
    save_cache,
    cached_score,
    has_valid_cache_entry,
    make_cache_entry,
)
import styles


class StepIndicator(QWidget):
    """Horizontal stepper: numbered dots joined by a track, with labels.

    The wizard is a genuine sequence (Select -> Analyze -> Filter), so numbered
    markers carry real meaning here. Completed steps fill in, the active step is
    accented, and upcoming steps stay muted.
    """

    def __init__(self, labels: List[str]):
        super().__init__()
        self.labels = labels
        self.current = 0
        self.setFixedHeight(64)
        self.setMinimumWidth(320)

    def set_step(self, index: int):
        self.current = index
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        n = len(self.labels)
        if n == 0:
            return

        radius = 15
        top = 6
        # Evenly space dot centers across the width, inset so labels don't clip.
        inset = 60
        usable = max(1, self.width() - 2 * inset)
        centers = [inset + (usable * i / (n - 1)) if n > 1 else self.width() / 2
                   for i in range(n)]
        cy = top + radius

        # Connecting tracks (drawn first, behind the dots)
        for i in range(n - 1):
            x1, x2 = centers[i] + radius, centers[i + 1] - radius
            done = i < self.current
            painter.setPen(QPen(
                QColor(styles.Colors.PRIMARY if done else styles.Colors.OUTLINE_VARIANT),
                3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(int(x1), int(cy), int(x2), int(cy))

        num_font = QFont(painter.font())
        num_font.setPointSize(11)
        num_font.setBold(True)

        label_font = QFont(painter.font())
        label_font.setPointSize(10)

        for i, (cx, label) in enumerate(zip(centers, self.labels)):
            completed = i < self.current
            active = i == self.current

            if completed:
                fill, ring, text = styles.Colors.PRIMARY, styles.Colors.PRIMARY, styles.Colors.ON_PRIMARY
            elif active:
                fill, ring, text = styles.Colors.PRIMARY_CONTAINER, styles.Colors.PRIMARY, styles.Colors.ON_PRIMARY_CONTAINER
            else:
                fill, ring, text = styles.Colors.SURFACE, styles.Colors.OUTLINE_VARIANT, styles.Colors.ON_SURFACE_VARIANT

            painter.setBrush(QBrush(QColor(fill)))
            painter.setPen(QPen(QColor(ring), 2))
            painter.drawEllipse(QPointF(cx, cy), radius, radius)

            painter.setFont(num_font)
            painter.setPen(QColor(text))
            dot_rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
            # A check for completed steps, the number otherwise.
            painter.drawText(dot_rect, Qt.AlignmentFlag.AlignCenter,
                             "✓" if completed else str(i + 1))

            painter.setFont(label_font)
            painter.setPen(QColor(styles.Colors.ON_SURFACE if active else styles.Colors.ON_SURFACE_VARIANT))
            label_rect = QRectF(cx - inset, cy + radius + 4, inset * 2, 18)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)


class HistogramWidget(QWidget):
    """Custom histogram widget for displaying blur score distribution."""
    bin_clicked = pyqtSignal(list, int) # Emits (list of items, bin_index)

    def __init__(self):
        super().__init__()
        self.scores = []
        self.results_data = {} # path -> score
        self.bins_data = [] # List of lists of (path, score)
        self.rects = [] # List of (rect, bin_index)
        self.selected_bin_index = -1
        self.threshold: Optional[float] = None  # score split: below = blurry
        self.setMinimumHeight(240)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_data(self, results: Dict[str, float]):
        """Update histogram with new results."""
        self.results_data = results
        self.scores = [s for s in results.values() if s is not None]
        self.update()

    def set_threshold(self, threshold: Optional[float]):
        """Set the blur threshold so bars recolor into blurry vs sharp."""
        self.threshold = threshold
        self.update()
    
    def paintEvent(self, event):
        """Draw the histogram, coloring bars blurry vs sharp around the threshold."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background / frame
        painter.fillRect(self.rect(), QColor(styles.Colors.SURFACE))
        painter.setPen(QPen(QColor(styles.Colors.OUTLINE_VARIANT), 1))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 12, 12)

        if not self.scores:
            painter.setPen(QColor(styles.Colors.ON_SURFACE_VARIANT))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Run analysis to see the score distribution")
            return

        # Fixed bin width of 5, aligned to multiples of 5
        import math
        lo = float(min(self.scores))
        hi = float(max(self.scores))
        if hi == lo:
            hi += 5.0  # give a degenerate distribution a small range

        bin_width = 5.0
        start_val = math.floor(lo / 5.0) * 5.0
        end_val = math.ceil(hi / 5.0) * 5.0

        bins = int((end_val - start_val) / bin_width)
        if bins < 1:
            bins = 1

        self.bins_data = [[] for _ in range(bins)]
        counts = [0] * bins

        for path, score in self.results_data.items():
            if score is None:
                continue
            if score < start_val:
                continue
            bin_idx = int((score - start_val) / bin_width)
            if bin_idx >= bins:
                bin_idx = bins - 1
            counts[bin_idx] += 1
            self.bins_data[bin_idx].append((path, score))

        # Plot geometry
        left_margin = 44
        right_margin = 20
        top_margin = 18
        bottom_margin = 52

        width = self.width() - left_margin - right_margin
        height = self.height() - top_margin - bottom_margin
        x_start = left_margin
        y_start = top_margin

        max_count = max(counts) if counts else 1
        bar_width = width / bins if bins > 0 else width

        # Gridlines — cap tick count so labels stay distinct integers
        painter.setPen(QPen(QColor(styles.Colors.SURFACE_VARIANT), 1, Qt.PenStyle.DotLine))
        n_yticks = min(4, max(1, max_count))
        for t in range(n_yticks + 1):
            y = y_start + height - (t / n_yticks) * height
            painter.drawLine(x_start, int(y), x_start + width, int(y))

        # Bars — colored by which side of the threshold the bin sits on.
        painter.setPen(Qt.PenStyle.NoPen)
        self.rects = []
        for i, count in enumerate(counts):
            bar_height = (count / max_count) * height if max_count > 0 else 0
            if count > 0:
                bar_height = max(bar_height, 3)  # keep single-count bins visible
            x = x_start + i * bar_width
            y = y_start + height - bar_height

            bin_start = start_val + i * bin_width
            bin_end = bin_start + bin_width
            base_color = self._bin_color(bin_start, bin_end)

            if i == self.selected_bin_index:
                painter.setBrush(QColor(base_color))
                painter.setPen(QPen(QColor(styles.Colors.ON_SURFACE), 2))
            else:
                painter.setBrush(QColor(base_color))
                painter.setPen(Qt.PenStyle.NoPen)

            rect = QRectF(x + 1, y, max(1, bar_width - 2), bar_height)
            painter.drawRoundedRect(rect, 3, 3)
            painter.setPen(Qt.PenStyle.NoPen)

            hit_rect = QRectF(x, y_start, bar_width, height)
            self.rects.append((hit_rect, i))

        # Threshold marker line
        if self.threshold is not None and end_val > start_val:
            if start_val <= self.threshold <= end_val:
                tx = x_start + (self.threshold - start_val) / (end_val - start_val) * width
                painter.setPen(QPen(QColor(styles.Colors.ON_SURFACE), 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(tx), y_start, int(tx), y_start + height)
                painter.setPen(QColor(styles.Colors.ON_SURFACE))
                marker_font = painter.font()
                marker_font.setPointSize(9)
                marker_font.setBold(True)
                painter.setFont(marker_font)
                painter.drawText(int(tx) - 40, y_start - 2, 80, 16,
                                 Qt.AlignmentFlag.AlignCenter, f"{self.threshold:.0f}")

        # Y-axis tick labels
        painter.setPen(QColor(styles.Colors.ON_SURFACE_VARIANT))
        font = painter.font()
        font.setPointSize(9)
        font.setBold(False)
        painter.setFont(font)
        for t in range(n_yticks + 1):
            val = int(round(t * (max_count / n_yticks)))
            y = y_start + height - (t / n_yticks) * height
            painter.drawText(0, int(y - 6), x_start - 6, 12, Qt.AlignmentFlag.AlignRight, str(val))

        # X-axis range labels
        painter.drawText(x_start, y_start + height + 4, 60, 16,
                         Qt.AlignmentFlag.AlignLeft, f"{start_val:.0f}")
        painter.drawText(x_start + width - 60, y_start + height + 4, 60, 16,
                         Qt.AlignmentFlag.AlignRight, f"{end_val:.0f}")

        # Legend / directional axis title: blurrier on the left, sharper on the right
        legend_y = y_start + height + 24
        painter.setPen(QColor(styles.Colors.BLURRY))
        painter.drawText(x_start, legend_y, width // 2, 18,
                         Qt.AlignmentFlag.AlignLeft, "← blurrier")
        painter.setPen(QColor(styles.Colors.SHARP))
        painter.drawText(x_start + width // 2, legend_y, width // 2, 18,
                         Qt.AlignmentFlag.AlignRight, "sharper →")
        painter.setPen(QColor(styles.Colors.ON_SURFACE_VARIANT))
        painter.drawText(x_start, legend_y, width, 18,
                         Qt.AlignmentFlag.AlignCenter, "Blur score (Laplacian variance)")

    def _bin_color(self, bin_start: float, bin_end: float) -> str:
        """Pick a bar color based on the threshold split."""
        if self.threshold is None:
            return styles.Colors.PRIMARY
        if bin_end <= self.threshold:
            return styles.Colors.BLURRY
        if bin_start >= self.threshold:
            return styles.Colors.SHARP
        # Straddles the threshold — lean by which side holds more of the bin.
        midpoint = (bin_start + bin_end) / 2
        return styles.Colors.BLURRY if midpoint < self.threshold else styles.Colors.SHARP

    def mousePressEvent(self, event):
        """Handle mouse clicks to select bins."""
        pos = event.position()
        clicked_index = -1
        for rect, idx in self.rects:
            if rect.contains(pos):
                clicked_index = idx
                break
        
        if clicked_index != -1:
            self.selected_bin_index = clicked_index
            self.update() # Redraw to show highlight
            if clicked_index < len(self.bins_data):
                self.bin_clicked.emit(self.bins_data[clicked_index], clicked_index)
        
        super().mousePressEvent(event)


def process_image_worker(image_path: str) -> Tuple[str, Optional[float]]:
    """Module-level function for processing images in multiprocessing pool."""
    return calculate_blurriness(image_path)


def load_qimage(path: str) -> Optional[QImage]:
    """Load any supported image (including RAW) into a QImage, or None on failure.

    Standard formats are read directly by QImage; RAW files are decoded via the
    shared ``blur_core._decode_image`` (BGR ndarray) and converted to a QImage.
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in RAW_EXTENSIONS:
            bgr = _decode_image(path)
            if bgr is None:
                return None
            # BGR -> RGB, then wrap the buffer in a QImage (copy to own the data).
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            h, w = rgb.shape[:2]
            img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            return img.copy()
        img = QImage(path)
        return None if img.isNull() else img
    except Exception:
        return None


class FullImageWorker(QThread):
    """Background worker that loads a single full-resolution image as a QImage."""
    image_ready = pyqtSignal(str, QImage)
    failed = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        img = load_qimage(self.path)
        if img is None:
            self.failed.emit(self.path)
        else:
            self.image_ready.emit(self.path, img)


class ImagePreviewDialog(QDialog):
    """Modal preview of a single image with Fit / 100% (actual-pixel) views.

    Blur is only reliably judgeable at native resolution, so the 100% mode shows
    an actual-pixel, scrollable view. The full image is loaded off the UI thread
    (RAW included) so opening the dialog never blocks.
    """

    def __init__(self, path: str, score: Optional[float], parent=None):
        super().__init__(parent)
        self.path = path
        self.full_image: Optional[QImage] = None
        self.fit_mode = True

        base = os.path.basename(path)
        score_txt = f"{score:.1f}" if score is not None else "—"
        self.setWindowTitle(f"{base}  ·  score {score_txt}")
        self.resize(900, 720)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Toolbar
        bar = QHBoxLayout()
        self.toggle_btn = QPushButton("View: Fit")
        self.toggle_btn.setProperty("role", "secondary")
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.clicked.connect(self.toggle_view)
        bar.addWidget(self.toggle_btn)

        self.reveal_btn = QPushButton("Reveal in Finder")
        self.reveal_btn.setProperty("role", "secondary")
        self.reveal_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reveal_btn.clicked.connect(self.reveal_in_file_manager)
        bar.addWidget(self.reveal_btn)

        bar.addStretch()
        self.caption = QLabel(f"{base}    ·    score {score_txt}")
        self.caption.setProperty("role", "caption")
        bar.addWidget(self.caption)
        layout.addLayout(bar)

        # Scrollable image area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.image_label = QLabel("Loading…")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(f"color: {styles.Colors.ON_SURFACE_VARIANT};")
        self.scroll.setWidget(self.image_label)
        layout.addWidget(self.scroll)

        # Kick off background load
        self._worker = FullImageWorker(path)
        self._worker.image_ready.connect(self._on_image_ready)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_image_ready(self, path: str, image: QImage):
        self.full_image = image
        self._render()

    def _on_failed(self, path: str):
        self.image_label.setText("Could not load this image.")

    def toggle_view(self):
        self.fit_mode = not self.fit_mode
        self.toggle_btn.setText("View: Fit" if self.fit_mode else "View: 100%")
        self.scroll.setWidgetResizable(self.fit_mode)
        self._render()

    def _render(self):
        """Paint the current image according to the active view mode."""
        if self.full_image is None:
            return
        pix = QPixmap.fromImage(self.full_image)
        if self.fit_mode:
            # Scale to the viewport, preserving aspect ratio.
            target = self.scroll.viewport().size()
            scaled = pix.scaled(
                target, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled)
            self.image_label.resize(target)
        else:
            # Actual pixels — let the scroll area handle overflow.
            self.image_label.setPixmap(pix)
            self.image_label.resize(pix.size())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit_mode:
            self._render()

    def reveal_in_file_manager(self):
        """Reveal the file in the OS file manager (the old default behavior)."""
        import subprocess
        import platform
        try:
            abs_path = os.path.abspath(self.path)
            system = platform.system()
            if system == "Darwin":
                subprocess.Popen(["open", "-R", abs_path])
            elif system == "Windows":
                subprocess.Popen(f'explorer /select, "{abs_path}"')
            elif system == "Linux":
                subprocess.Popen(["xdg-open", os.path.dirname(abs_path)])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to reveal file: {e}")


class ThumbnailWorker(QThread):
    """Background worker to generate QImage thumbnails for file paths."""
    thumbnail_ready = pyqtSignal(str, QImage)

    def __init__(self, paths: List[str], size: int = 128):
        super().__init__()
        self.paths = paths
        self.size = size

    def run(self):
        for p in self.paths:
            try:
                if not os.path.isfile(p):
                    continue
                img = QImage(p)
                if img.isNull():
                    continue
                thumb = img.scaled(self.size, self.size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.thumbnail_ready.emit(p, thumb)
            except Exception:
                # ignore individual failures
                continue


class AnalysisThread(QThread):
    """Worker thread for image analysis.

    Only paths not already satisfied by the cache are sent through the pool;
    cached results are passed in and merged into the emitted dict. Setting
    ``cancel()`` terminates the pool and emits ``cancelled``.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, to_compute: List[str], cached_results: Dict[str, Optional[float]],
                 cache_file: str, cache_entries: Dict[str, dict]):
        super().__init__()
        self.to_compute = to_compute
        self.cached_results = cached_results
        self.cache_file = cache_file
        self.cache_entries = cache_entries
        self._cancelled = False

    def cancel(self):
        """Request cancellation; the run loop terminates the pool at the next result."""
        self._cancelled = True

    def run(self):
        """Run analysis in background, streaming progress and updating the cache."""
        try:
            results: Dict[str, Optional[float]] = dict(self.cached_results)
            total = len(self.to_compute)

            if total > 0:
                num_processes = cpu_count()
                with Pool(processes=num_processes) as pool:
                    it = pool.imap_unordered(process_image_worker, self.to_compute)
                    for idx, (path, score) in enumerate(it):
                        if self._cancelled:
                            pool.terminate()
                            pool.join()
                            save_cache(self.cache_file, self.cache_entries)
                            self.cancelled.emit()
                            return
                        results[path] = score
                        entry = make_cache_entry(path, score)
                        if entry is not None:
                            self.cache_entries[path] = entry
                        progress = int((idx + 1) / total * 100)
                        self.progress.emit(progress)
            else:
                self.progress.emit(100)

            save_cache(self.cache_file, self.cache_entries)
            self.finished.emit(results)
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            self.error.emit(error_msg)


class BlurDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blur Detector - Advanced Image Analysis")
        self.setGeometry(100, 100, 1000, 800)
        
        # Data storage
        self.selected_paths: List[str] = []
        self.image_paths_to_process: List[str] = []
        self.results: Dict[str, Optional[float]] = {}
        self.blurry_paths: List[str] = []
        self.analysis_thread = None
        self._thumb_workers: List[ThumbnailWorker] = []
        self.current_step = 0
        
        # Setup UI
        self.setup_ui()
        self.setStyleSheet(styles.get_stylesheet())
        self.show_step(0)
    
    def setup_ui(self):
        """Create the main UI with stacked pages."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(20)
        
        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        title = QLabel("Blur Detector")
        title.setProperty("role", "display")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)

        subtitle = QLabel("Find soft shots with Laplacian variance, then move or delete them")
        subtitle.setProperty("role", "subheading")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(subtitle)

        # Stepper (replaces the old "Step N of 3" text)
        self.step_indicator = StepIndicator(["Select", "Analyze", "Filter"])
        stepper_row = QHBoxLayout()
        stepper_row.addStretch()
        stepper_row.addWidget(self.step_indicator)
        stepper_row.addStretch()
        header_layout.addSpacing(8)
        header_layout.addLayout(stepper_row)

        main_layout.addLayout(header_layout)
        
        # Main Card Container
        card = QFrame()
        card.setProperty("role", "card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.setSpacing(20)
        
        # Divider (Removed)
        # line = QFrame()
        # line.setFrameShape(QFrame.Shape.HLine)
        # line.setFrameShadow(QFrame.Shadow.Sunken)
        # line.setStyleSheet(f"color: {styles.Colors.OUTLINE};")
        # card_layout.addWidget(line)

        # Scroll Area for Content
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Stacked widget for steps
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.create_step_select_images())
        self.stacked_widget.addWidget(self.create_step_analyze())
        self.stacked_widget.addWidget(self.create_step_filter())
        
        self.scroll_area.setWidget(self.stacked_widget)
        card_layout.addWidget(self.scroll_area)
        
        # Navigation Buttons
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("Back")
        self.prev_btn.setProperty("role", "secondary")
        self.prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_btn.clicked.connect(self.prev_step)
        nav_layout.addWidget(self.prev_btn)
        
        nav_layout.addStretch()
        
        # Action Buttons (Hidden by default, shown in Step 3)
        self.delete_btn = QPushButton("Trash Blurry")
        self.delete_btn.setProperty("role", "danger")
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.clicked.connect(self.delete_blurry)
        self.delete_btn.hide()
        nav_layout.addWidget(self.delete_btn)
        
        self.move_btn = QPushButton("Move Blurry")
        self.move_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.move_btn.clicked.connect(self.move_blurry)
        self.move_btn.hide()
        nav_layout.addWidget(self.move_btn)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.clicked.connect(self.next_step)
        nav_layout.addWidget(self.next_btn)
        
        card_layout.addLayout(nav_layout)
        
        main_layout.addWidget(card)
    
    def show_step(self, step: int):
        """Show a specific step."""
        self.current_step = step
        self.stacked_widget.setCurrentIndex(step)
        self.step_indicator.set_step(step)
        
        # Update button visibility
        self.prev_btn.setEnabled(step > 0)
        
        # Reset button visibility
        self.next_btn.hide()
        self.delete_btn.hide()
        self.move_btn.hide()
        
        if step == 0:
            self.next_btn.setText("Next")
            self.next_btn.show()
        elif step == 2: # Filter step
            self.delete_btn.show()
            self.move_btn.show()
            # Offer a data-driven default: if the field is still the initial
            # placeholder, pre-fill it with the Otsu suggestion.
            self.refresh_suggestion_label()
            if self.threshold_input.text().strip() in ("", "100.0"):
                suggestion = self.suggested_threshold()
                if suggestion is not None:
                    self.threshold_input.setText(f"{suggestion:.1f}")
            # Seed the histogram split with the current threshold value
            self.on_threshold_changed(self.threshold_input.text())
        else:
            self.next_btn.setText("Next")
            self.next_btn.show()
    
    def prev_step(self):
        """Go to previous step."""
        if self.current_step > 0:
            self.show_step(self.current_step - 1)
    
    def next_step(self):
        """Go to next step."""
        if self.current_step == 0:  # Select Images
            if not self.selected_paths:
                QMessageBox.warning(self, "Warning", "Please select at least one image or folder!")
                return
        elif self.current_step == 1:  # Analyze
            if not self.results:
                QMessageBox.warning(self, "Warning", "Please run analysis first!")
                return
        elif self.current_step == 2:  # Filter
            pass
        
        if self.current_step < 2:
            self.show_step(self.current_step + 1)
    
    def create_step_select_images(self) -> QWidget:
        """Create Step 1: Select Images."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label = QLabel("Select Images or Folders")
        label.setProperty("role", "heading")
        layout.addWidget(label)
        
        desc = QLabel("Choose image files or folders to analyze:")
        desc.setProperty("role", "subheading")
        layout.addWidget(desc)
        
        layout.addSpacing(20)
        
        btn_layout = QHBoxLayout()
        btn_files = QPushButton("Add Images")
        btn_files.setIcon(QIcon.fromTheme("image-x-generic")) # Optional: use system icons if available
        btn_files.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_files.clicked.connect(self.add_image_files)
        btn_layout.addWidget(btn_files)
        
        btn_folder = QPushButton("Add Folder")
        btn_folder.setIcon(QIcon.fromTheme("folder"))
        btn_folder.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_folder.clicked.connect(self.add_folder)
        btn_layout.addWidget(btn_folder)
        
        btn_clear = QPushButton("Clear")
        btn_clear.setProperty("role", "secondary")
        btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear.clicked.connect(self.clear_selection)
        btn_layout.addWidget(btn_clear)
        
        layout.addLayout(btn_layout)
        
        self.path_label = QLabel("No images or folders selected yet")
        self.path_label.setStyleSheet(f"color: {styles.Colors.ON_SURFACE_VARIANT}; font-style: italic; margin-top: 10px;")
        layout.addWidget(self.path_label)

        hint = QLabel("Tip: select an item below and press Delete to remove just that one.")
        hint.setProperty("role", "caption")
        layout.addWidget(hint)

        # Gallery of selected items
        self.gallery_list = QListWidget()
        self.gallery_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery_list.setIconSize(QPixmap(100,100).size())
        self.gallery_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gallery_list.setMovement(QListWidget.Movement.Static)
        self.gallery_list.setSpacing(12)
        self.gallery_list.setWrapping(True)
        self.gallery_list.setMinimumHeight(250)
        self.gallery_list.keyPressEvent = self._gallery_key_press
        layout.addWidget(self.gallery_list)

        return widget

    def _gallery_key_press(self, event):
        """Remove the selected gallery item(s) on Delete/Backspace."""
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            removed = False
            for item in self.gallery_list.selectedItems():
                path = item.data(Qt.ItemDataRole.UserRole)
                if path in self.selected_paths:
                    self.selected_paths.remove(path)
                    removed = True
            if removed:
                self.update_path_label()
                self.update_gallery()
                return
        QListWidget.keyPressEvent(self.gallery_list, event)
    
    def create_step_analyze(self) -> QWidget:
        """Create Step 2: Analyze Images."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label = QLabel("Analyze Images")
        label.setProperty("role", "heading")
        layout.addWidget(label)
        
        desc = QLabel("Score every image, then explore the distribution below:")
        desc.setProperty("role", "subheading")
        layout.addWidget(desc)
        
        layout.addSpacing(20)
        
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Analysis")
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self.start_analysis)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setProperty("role", "danger")
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.hide()
        btn_layout.addWidget(self.stop_btn)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: {styles.Colors.PRIMARY}; font-weight: bold; margin-left: 20px;")
        btn_layout.addWidget(self.status_label)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        layout.addSpacing(20)
        
        # Histogram widget
        self.histogram_widget = HistogramWidget()
        self.histogram_widget.bin_clicked.connect(self.on_bin_clicked)
        layout.addWidget(self.histogram_widget)
        
        # Bin details list (replaces sample label)
        self.bin_info_label = QLabel("Click a bar to list the images in that score range.")
        self.bin_info_label.setStyleSheet(f"color: {styles.Colors.ON_SURFACE_VARIANT}; font-style: italic; margin-top: 10px;")
        layout.addWidget(self.bin_info_label)

        self.bin_details_list = QListWidget()
        self.bin_details_list.setViewMode(QListWidget.ViewMode.ListMode)
        self.bin_details_list.setIconSize(QPixmap(64,64).size())
        self.bin_details_list.setSpacing(5)
        self.bin_details_list.setMinimumHeight(200)
        self.bin_details_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        layout.addWidget(self.bin_details_list)
        
        return widget
    
    def create_step_filter(self) -> QWidget:
        """Create Step 3: Filter Results."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label = QLabel("Filter Results")
        label.setProperty("role", "heading")
        layout.addWidget(label)
        
        desc = QLabel("Set a threshold — anything below it counts as blurry:")
        desc.setProperty("role", "subheading")
        layout.addWidget(desc)

        layout.addSpacing(20)

        filter_layout = QHBoxLayout()
        threshold_label = QLabel("Threshold:")
        filter_layout.addWidget(threshold_label)

        self.threshold_input = QLineEdit("100.0")
        self.threshold_input.setMaximumWidth(100)
        self.threshold_input.textChanged.connect(self.on_threshold_changed)
        filter_layout.addWidget(self.threshold_input)

        btn = QPushButton("Filter & Show Results")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(self.filter_results)
        filter_layout.addWidget(btn)

        self.suggest_btn = QPushButton("Use suggested")
        self.suggest_btn.setProperty("role", "secondary")
        self.suggest_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.suggest_btn.clicked.connect(self.apply_suggested_threshold)
        filter_layout.addWidget(self.suggest_btn)

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        self.suggestion_label = QLabel("")
        self.suggestion_label.setProperty("role", "caption")
        layout.addWidget(self.suggestion_label)

        self.filter_summary = QLabel("")
        self.filter_summary.setProperty("role", "caption")
        layout.addWidget(self.filter_summary)

        # List of filtered images (click to preview)
        self.filtered_list = QListWidget()
        self.filtered_list.setMinimumHeight(300)
        self.filtered_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        layout.addWidget(self.filtered_list)

        # Open button
        open_layout = QHBoxLayout()
        self.open_btn = QPushButton("Open Selected")
        self.open_btn.setProperty("role", "secondary")
        self.open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_btn.clicked.connect(self.open_selected_image)
        open_layout.addWidget(self.open_btn)
        open_layout.addStretch()
        layout.addLayout(open_layout)
        
        layout.addSpacing(20)
        
        # Actions (Moved to footer)
        # action_label = QLabel("Actions:")
        # action_label.setProperty("role", "subheading")
        # layout.addWidget(action_label)
        
        # action_layout = QHBoxLayout()
        
        # delete_btn = QPushButton("Delete Blurry Images")
        # delete_btn.setProperty("role", "danger")
        # delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # delete_btn.clicked.connect(self.delete_blurry)
        # action_layout.addWidget(delete_btn)
        
        # move_btn = QPushButton("Move Blurry Images")
        # move_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # move_btn.clicked.connect(self.move_blurry)
        # action_layout.addWidget(move_btn)
        
        # action_layout.addStretch()
        # layout.addLayout(action_layout)
        
        return widget
    
    def add_image_files(self):
        """Open a multi-file dialog to add image files to selection."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select image files",
            filter="Images (*.jpg *.jpeg *.png *.gif *.bmp *.tiff *.webp *.cr3 *.dng *.nef *.arw);;All Files (*)"
        )
        if files:
            # append without duplicates while preserving order
            for f in files:
                if f not in self.selected_paths:
                    self.selected_paths.append(f)
            self.update_path_label()
            self.update_gallery()

    def add_folder(self):
        """Open folder selection dialog."""
        folder = QFileDialog.getExistingDirectory(self, "Select folder containing images")
        if folder:
            if folder not in self.selected_paths:
                self.selected_paths.append(folder)
            self.update_path_label()
            self.update_gallery()

    def clear_selection(self):
        """Clear all selected paths."""
        self.selected_paths = []
        self.update_path_label()
        self.update_gallery()
    
    def update_path_label(self):
        """Update selected paths display."""
        if self.selected_paths:
            # show up to 5 entries (files show basename)
            display_items = []
            for p in self.selected_paths[:5]:
                if os.path.isfile(p):
                    display_items.append(f"• {os.path.basename(p)}")
                else:
                    display_items.append(f"• {p}")
            text = "\n".join(display_items)
            if len(self.selected_paths) > 5:
                text += f"\n... and {len(self.selected_paths) - 5} more"
            self.path_label.setText(text)
            self.path_label.setStyleSheet(f"color: {styles.Colors.SHARP}; font-weight: 600;")
        else:
            self.path_label.setText("No images or folders selected yet")
            self.path_label.setStyleSheet(f"color: {styles.Colors.ON_SURFACE_VARIANT}; font-style: italic;")
        # also refresh gallery
        try:
            self.update_gallery()
        except Exception:
            pass
    
    def update_filtered_list(self):
        """Update the Step 3 list with filtered blurry images, showing scores."""
        self.filtered_list.clear()
        for p in self.blurry_paths:
            score = self.results.get(p)
            score_txt = f"{score:.1f}" if score is not None else "—"
            item = QListWidgetItem(f"⬤  {os.path.basename(p)}    ·    {score_txt}")
            item.setForeground(QColor(styles.Colors.BLURRY))
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.filtered_list.addItem(item)

    def on_threshold_changed(self, text: str):
        """Recolor the histogram split as the threshold field changes."""
        if not hasattr(self, 'histogram_widget'):
            return
        try:
            value = float(text)
        except ValueError:
            self.histogram_widget.set_threshold(None)
            return
        self.histogram_widget.set_threshold(value)

    def suggested_threshold(self) -> Optional[float]:
        """Suggest a blurry/sharp split using Otsu's method on the score histogram.

        Otsu finds the threshold that maximizes between-class variance — a natural
        valley between a "blurry" cluster and a "sharp" cluster. Returns ``None``
        when there aren't enough distinct scores to split meaningfully.
        """
        scores = np.array([s for s in self.results.values() if s is not None], dtype=float)
        if scores.size < 2:
            return None
        lo, hi = float(scores.min()), float(scores.max())
        if hi <= lo:
            return None

        # 256-bin histogram, then classic Otsu over the bins.
        n_bins = 256
        hist, edges = np.histogram(scores, bins=n_bins, range=(lo, hi))
        total = scores.size
        centers = (edges[:-1] + edges[1:]) / 2.0

        weight_bg = np.cumsum(hist)
        weight_fg = total - weight_bg
        # Guard against empty classes.
        valid = (weight_bg > 0) & (weight_fg > 0)
        if not valid.any():
            return None

        cum_mean = np.cumsum(hist * centers)
        mean_bg = np.where(weight_bg > 0, cum_mean / np.maximum(weight_bg, 1), 0.0)
        total_mean = cum_mean[-1]
        mean_fg = np.where(weight_fg > 0, (total_mean - cum_mean) / np.maximum(weight_fg, 1), 0.0)

        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        between[~valid] = -1.0
        best = int(np.argmax(between))
        return float(centers[best])

    def apply_suggested_threshold(self):
        """Fill the threshold field with the Otsu suggestion, if available."""
        value = self.suggested_threshold()
        if value is None:
            QMessageBox.information(self, "No suggestion",
                                    "Not enough score variation to suggest a threshold.")
            return
        self.threshold_input.setText(f"{value:.1f}")

    def refresh_suggestion_label(self):
        """Update the 'Suggested: N' caption on the filter step."""
        if not hasattr(self, 'suggestion_label'):
            return
        value = self.suggested_threshold()
        if value is None:
            self.suggestion_label.setText("")
            self.suggest_btn.setEnabled(False)
        else:
            self.suggestion_label.setText(f"Suggested threshold: {value:.1f} (Otsu split of the distribution)")
            self.suggest_btn.setEnabled(True)

    def update_gallery(self):
        """Update the Step 1 gallery showing selected files/folders."""
        if not hasattr(self, 'gallery_list'):
            return
        self.gallery_list.clear()
        files_to_thumb: List[str] = []
        # Build items with placeholders, and collect files that need thumbnails
        for p in self.selected_paths:
            base = os.path.basename(p)
            item = QListWidgetItem(base)
            item.setData(Qt.ItemDataRole.UserRole, p)

            if os.path.isfile(p):
                # light placeholder until the thumbnail arrives
                pm = QPixmap(128, 128)
                pm.fill(QColor(styles.Colors.SURFACE_CONTAINER))
                painter = QPainter(pm)
                painter.setPen(QPen(QColor(styles.Colors.ON_SURFACE_VARIANT)))
                painter.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, base[:15])
                painter.end()
                item.setIcon(QIcon(pm))
                files_to_thumb.append(p)
            else:
                # folder placeholder — dashed outline to read as a container
                pm = QPixmap(128, 128)
                pm.fill(QColor(styles.Colors.PRIMARY_CONTAINER))
                painter = QPainter(pm)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setPen(QPen(QColor(styles.Colors.PRIMARY), 2, Qt.PenStyle.DashLine))
                painter.drawRoundedRect(6, 6, 115, 115, 8, 8)
                painter.setPen(QPen(QColor(styles.Colors.ON_PRIMARY_CONTAINER)))
                painter.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, base[:20])
                painter.end()
                item.setIcon(QIcon(pm))

            self.gallery_list.addItem(item)

        # Start a background thumbnail worker for file paths if any
        if files_to_thumb:
            worker = ThumbnailWorker(files_to_thumb, size=128)
            worker.thumbnail_ready.connect(self.on_thumbnail_ready)
            # keep reference so worker is not garbage-collected
            self._thumb_workers.append(worker)

            def _on_finished(w=worker):
                try:
                    self._thumb_workers.remove(w)
                except ValueError:
                    pass

            worker.finished.connect(_on_finished)
            worker.start()

    def on_thumbnail_ready(self, path: str, qimage: QImage):
        """Slot called when a thumbnail QImage is ready for a path."""
        # Find the corresponding QListWidgetItem and set its icon
        for i in range(self.gallery_list.count()):
            item = self.gallery_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == path:
                try:
                    pix = QPixmap.fromImage(qimage)
                    icon = QIcon(pix)
                    item.setIcon(icon)
                except Exception:
                    pass
                break

    def open_selected_image(self):
        """Open the selected image in a preview dialog."""
        item = self.filtered_list.currentItem()
        if not item:
            QMessageBox.warning(self, "Warning", "No image selected!")
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path or not os.path.exists(path):
            QMessageBox.critical(self, "Error", "Image not found on disk.")
            return
        self.show_image_preview(path)

    def on_item_double_clicked(self, item: QListWidgetItem):
        """Handle double-click on list item to preview image."""
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and os.path.exists(path):
            self.show_image_preview(path)

    def show_image_preview(self, path: str):
        """Open the image in an in-app preview dialog (Fit / 100% views)."""
        if not os.path.exists(path):
            QMessageBox.critical(self, "Error", "Image not found on disk.")
            return
        score = self.results.get(path)
        dialog = ImagePreviewDialog(path, score, parent=self)
        dialog.exec()
    
    def start_analysis(self):
        """Start image analysis."""
        if not self.selected_paths:
            QMessageBox.warning(self, "Warning", "Please select files or folders first!")
            return
        
        if self.analysis_thread and self.analysis_thread.isRunning():
            QMessageBox.warning(self, "Warning", "Analysis already in progress!")
            return
        
        # Collect all images
        self.image_paths_to_process = []
        for path in self.selected_paths:
            self.image_paths_to_process.extend(get_image_paths(path))
        
        self.image_paths_to_process = list(dict.fromkeys(self.image_paths_to_process))
        
        if not self.image_paths_to_process:
            QMessageBox.critical(self, "Error", "No images found in selected paths!")
            return

        # Split work using the cache: cached hits are reused, only misses run.
        cache_file = cache_path_for(self.selected_paths)
        cache_entries = load_cache(cache_file)
        cached_results: Dict[str, Optional[float]] = {}
        to_compute: List[str] = []
        for path in self.image_paths_to_process:
            if has_valid_cache_entry(cache_entries, path):
                cached_results[path] = cached_score(cache_entries, path)
            else:
                to_compute.append(path)

        # Start analysis thread
        cached_n = len(cached_results)
        if cached_n:
            self.status_label.setText(f"Analyzing… ({cached_n} cached)")
        else:
            self.status_label.setText("Analyzing…")
        self.status_label.setStyleSheet(f"color: {styles.Colors.WARNING}; font-weight: 600;")
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.show()

        self.analysis_thread = AnalysisThread(
            to_compute, cached_results, cache_file, cache_entries)
        self.analysis_thread.progress.connect(self.update_progress)
        self.analysis_thread.finished.connect(self.on_analysis_complete)
        self.analysis_thread.error.connect(self.on_analysis_error)
        self.analysis_thread.cancelled.connect(self.on_analysis_cancelled)
        self.analysis_thread.start()

    def stop_analysis(self):
        """Request cancellation of the running analysis."""
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.status_label.setText("Stopping…")
            self.analysis_thread.cancel()

    def _reset_analysis_controls(self):
        """Restore Start/Stop button state after analysis ends or is cancelled."""
        self.start_btn.setEnabled(True)
        self.stop_btn.hide()

    def update_progress(self, value: int):
        """Update progress bar."""
        self.progress_bar.setValue(value)

    def on_analysis_complete(self, results: Dict):
        """Handle analysis completion."""
        self.results = results
        n_failed = sum(1 for s in results.values() if s is None)
        n_ok = len(results) - n_failed
        if n_failed:
            self.status_label.setText(f"✓ Analyzed {n_ok}, {n_failed} unreadable")
        else:
            self.status_label.setText(f"✓ Analysis complete ({n_ok})")
        self.status_label.setStyleSheet(f"color: {styles.Colors.SUCCESS}; font-weight: bold;")
        self.histogram_widget.set_data(results)
        self._reset_analysis_controls()
        # self.display_results() # Removed as per request

    def on_analysis_cancelled(self):
        """Handle a user-cancelled analysis: keep whatever finished, reset UI."""
        self.status_label.setText("Analysis stopped")
        self.status_label.setStyleSheet(f"color: {styles.Colors.ON_SURFACE_VARIANT}; font-weight: 600;")
        self._reset_analysis_controls()
    
    def on_bin_clicked(self, items: List[Tuple[str, float]], bin_index: int):
        """Handle click on histogram bin."""
        self.bin_details_list.clear()
        
        # Calculate range
        # Assuming fixed bin width of 5.0 and start_val is aligned to 5.0
        # We need to know the start_val used in HistogramWidget. 
        # Ideally HistogramWidget should pass the range, but we can infer or just ask it.
        # For now, let's recalculate or approximate based on index if we knew the min.
        # Better: HistogramWidget logic uses start_val = math.floor(lo / 5.0) * 5.0
        # We can access self.histogram_widget.scores to re-derive start_val or just pass it.
        
        # Let's just calculate it here quickly using the same logic
        scores = [s for s in self.results.values() if s is not None]
        if not scores: return
        
        import math
        lo = float(min(scores))
        start_val = math.floor(lo / 5.0) * 5.0
        bin_start = start_val + (bin_index * 5.0)
        bin_end = bin_start + 5.0
        
        self.bin_info_label.setText(f"Range: {bin_start:.0f} - {bin_end:.0f} | Showing {len(items)} image(s):")
        
        files_to_thumb = []
        
        for path, score in items:
            item = QListWidgetItem(f"{os.path.basename(path)} (Score: {score:.2f})")
            item.setData(Qt.ItemDataRole.UserRole, path)
            
            # Placeholder icon
            pm = QPixmap(64, 64)
            pm.fill(QColor(styles.Colors.SURFACE_VARIANT))
            item.setIcon(QIcon(pm))
            
            self.bin_details_list.addItem(item)
            files_to_thumb.append(path)
            
        # Generate thumbnails for this list
        if files_to_thumb:
            worker = ThumbnailWorker(files_to_thumb, size=64)
            worker.thumbnail_ready.connect(self.on_bin_thumbnail_ready)
            self._thumb_workers.append(worker)
            
            def _on_finished(w=worker):
                try:
                    self._thumb_workers.remove(w)
                except ValueError:
                    pass
            
            worker.finished.connect(_on_finished)
            worker.start()

    def on_bin_thumbnail_ready(self, path: str, qimage: QImage):
        """Update thumbnail in bin details list."""
        for i in range(self.bin_details_list.count()):
            item = self.bin_details_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == path:
                try:
                    pix = QPixmap.fromImage(qimage)
                    item.setIcon(QIcon(pix))
                except Exception:
                    pass
                break

    def draw_histogram(self):
        """Draw blur score histogram in the analysis section."""
        # Handled by set_data now
        pass
    
    def on_analysis_error(self, error: str):
        """Handle analysis error."""
        self.status_label.setText("✗ Error")
        self.status_label.setStyleSheet(f"color: {styles.Colors.ERROR}; font-weight: bold;")
        self._reset_analysis_controls()
        QMessageBox.critical(self, "Analysis Error", error)
    
    def display_results(self):
        """Display analysis results."""
        # Deprecated/Removed
        pass
    
    def filter_results(self):
        """Filter results by threshold."""
        try:
            threshold = float(self.threshold_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid threshold value!")
            return
        
        if not self.results:
            QMessageBox.warning(self, "Warning", "Run analysis first!")
            return
        self.histogram_widget.set_threshold(threshold)
        self.blurry_paths = [p for p in self.results if self.results[p] is not None and self.results[p] < threshold]

        # Populate the filtered list widget with filenames and store full path in item data
        self.update_filtered_list()

        total = sum(1 for s in self.results.values() if s is not None)
        n_blur = len(self.blurry_paths)
        self.filter_summary.setText(
            f"{n_blur} of {total} image(s) fall below {threshold:.0f} and are flagged as blurry.")

        if not self.blurry_paths:
            QMessageBox.information(self, "No blurry images",
                                    "Nothing scored below this threshold. Try raising it.")
    
    def delete_blurry(self):
        """Send blurry images to the OS trash (recoverable), not a hard delete."""
        if not self.blurry_paths:
            QMessageBox.warning(self, "Warning", "Filter results first!")
            return

        reply = QMessageBox.question(
            self, "Confirm",
            f"Move {len(self.blurry_paths)} blurry image(s) to the Trash?\n"
            "You can restore them from the Trash if needed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Iterate resiliently: one failure shouldn't abort the rest.
        succeeded: List[str] = []
        failures: List[Tuple[str, str]] = []
        for path in self.blurry_paths:
            try:
                if os.path.exists(path):
                    send2trash(path)
                succeeded.append(path)
            except Exception as e:
                failures.append((path, str(e)))

        self._after_removal(succeeded)
        self._report_action("trashed", succeeded, failures)

    def move_blurry(self):
        """Move blurry images to a chosen folder, collision-safe and resilient."""
        if not self.blurry_paths:
            QMessageBox.warning(self, "Warning", "Filter results first!")
            return

        dest = QFileDialog.getExistingDirectory(self, "Select destination folder")
        if not dest:
            return

        try:
            os.makedirs(dest, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Could not create destination: {e}")
            return

        succeeded: List[str] = []
        failures: List[Tuple[str, str]] = []
        for path in self.blurry_paths:
            try:
                if not os.path.exists(path):
                    failures.append((path, "file no longer exists"))
                    continue
                destination = unique_destination(dest, os.path.basename(path))
                shutil.move(path, destination)
                succeeded.append(path)
            except Exception as e:
                failures.append((path, str(e)))

        self._after_removal(succeeded)
        self._report_action("moved", succeeded, failures)

    def _after_removal(self, processed: List[str]):
        """Drop processed paths from state and refresh the list + histogram."""
        if not processed:
            return
        processed_set = set(processed)
        self.blurry_paths = [p for p in self.blurry_paths if p not in processed_set]
        for p in processed:
            self.results.pop(p, None)
        self.update_filtered_list()
        self.histogram_widget.set_data(self.results)

    def _report_action(self, verb: str, succeeded: List[str],
                       failures: List[Tuple[str, str]]):
        """Show a summary of a move/trash action, including any per-file failures."""
        if failures:
            detail = "\n".join(f"• {os.path.basename(p)}: {err}" for p, err in failures[:10])
            if len(failures) > 10:
                detail += f"\n… and {len(failures) - 10} more"
            QMessageBox.warning(
                self, "Completed with errors",
                f"{len(succeeded)} image(s) {verb}, {len(failures)} failed:\n\n{detail}")
        else:
            QMessageBox.information(
                self, "Success", f"{len(succeeded)} image(s) {verb}.")


if __name__ == '__main__':
    # Required for PyInstaller-frozen builds on Windows (spawn start method):
    # without it each pool worker re-executes this block and re-launches the GUI.
    freeze_support()
    app = QApplication(sys.argv)
    window = BlurDetectorGUI()
    window.show()
    sys.exit(app.exec())
