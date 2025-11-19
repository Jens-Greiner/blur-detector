import sys
import threading
import os
import shutil
import traceback
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count

import rawpy
import numpy as np
import cv2

# On macOS Qt6 needs this environment variable set before importing PyQt
if sys.platform == 'darwin':
    os.environ.setdefault('QT_MAC_WANTS_LAYER', '1')

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog, QMessageBox,
    QProgressBar, QComboBox, QSpinBox, QFrame, QScrollArea, QListWidget,
    QListWidgetItem, QDialog, QRadioButton, QButtonGroup, QStackedWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF
from PyQt6.QtGui import QFont, QColor, QIcon, QPixmap, QPainter, QPen, QBrush, QImage
import styles


class HistogramWidget(QWidget):
    """Custom histogram widget for displaying blur score distribution."""
    bin_clicked = pyqtSignal(list, int) # Emits (list of items, bin_index)

    def __init__(self):
        super().__init__()
        self.scores = []
        self.results_data = {} # path -> score
        self.bins_data = [] # List of lists of (path, score)
        self.bins_data = [] # List of lists of (path, score)
        self.rects = [] # List of (rect, bin_index)
        self.selected_bin_index = -1
        self.setMinimumHeight(200)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # self.setStyleSheet("background-color: white; border: 1px solid #ddd; border-radius: 5px;") # Handled by paintEvent and parent style
    
    def set_data(self, results: Dict[str, float]):
        """Update histogram with new results."""
        self.results_data = results
        self.scores = [s for s in results.values() if s is not None]
        self.update()
    
    def paintEvent(self, event):
        """Draw the histogram."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background/frame
        painter.fillRect(self.rect(), QColor(styles.Colors.SURFACE))
        painter.setPen(QPen(QColor(styles.Colors.OUTLINE), 1))
        painter.drawRoundedRect(self.rect().adjusted(1,1,-1,-1), 4, 4)

        if not self.scores:
            painter.setPen(QColor(styles.Colors.ON_SURFACE_VARIANT))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No data available")
            return

        # Calculate histogram with fixed bin width of 5
        import math
        lo = float(min(self.scores))
        hi = float(max(self.scores))
        
        # Ensure we have a range
        if hi == lo:
            hi += 5.0 # Create a small range if all equal
            
        bin_width = 5.0
        # Align 'lo' to the nearest multiple of 5 below
        start_val = math.floor(lo / 5.0) * 5.0
        # Align 'hi' to the nearest multiple of 5 above
        end_val = math.ceil(hi / 5.0) * 5.0
        
        # Calculate number of bins
        bins = int((end_val - start_val) / bin_width)
        if bins < 1: bins = 1
        
        self.bins_data = [[] for _ in range(bins)]
        counts = [0] * bins
        
        # Populate bins
        for path, score in self.results_data.items():
            if score is None: continue
            if score < start_val: continue
            bin_idx = int((score - start_val) / bin_width)
            if bin_idx >= bins: bin_idx = bins - 1
            counts[bin_idx] += 1
            self.bins_data[bin_idx].append((path, score))

        # Draw histogram area
        left_margin = 50
        right_margin = 20
        top_margin = 20
        bottom_margin = 40

        width = self.width() - left_margin - right_margin
        height = self.height() - top_margin - bottom_margin
        x_start = left_margin
        y_start = top_margin

        max_count = max(counts) if counts else 1
        bar_width = width / bins if bins > 0 else width

        # Grid lines
        painter.setPen(QPen(QColor(styles.Colors.SURFACE_VARIANT), 1, Qt.PenStyle.DotLine))
        n_yticks = 5
        for t in range(n_yticks + 1):
            y = y_start + height - (t / n_yticks) * height
            painter.drawLine(x_start, int(y), x_start + width, int(y))

        # Bars
        painter.setPen(Qt.PenStyle.NoPen)
        
        self.rects = []
        for i, count in enumerate(counts):
            bar_height = (count / max_count) * height if max_count > 0 else 0
            x = x_start + i * bar_width
            y = y_start + height - bar_height
            
            # Set brush color (highlight selected)
            if i == self.selected_bin_index:
                painter.setBrush(QColor(styles.Colors.TERTIARY)) # Highlight color
            else:
                painter.setBrush(QColor(styles.Colors.PRIMARY))
            
            # Draw rounded top bars
            rect = QRectF(x, y, max(1, bar_width - 1), bar_height)
            painter.drawRoundedRect(rect, 2, 2)
            
            # Store rect for hit testing (expand width slightly for easier clicking)
            hit_rect = QRectF(x, y_start, bar_width, height)
            self.rects.append((hit_rect, i))

        # Axes and Labels
        painter.setPen(QPen(QColor(styles.Colors.ON_SURFACE_VARIANT)))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)

        # Y-axis labels
        for t in range(n_yticks + 1):
            val = int(round(t * (max_count / n_yticks)))
            y = y_start + height - (t / n_yticks) * height
            painter.drawText(0, int(y - 5), x_start - 5, 10, Qt.AlignmentFlag.AlignRight, str(val))

        # X-axis labels (min and max)
        painter.drawText(x_start, y_start + height + 5, 50, 20, Qt.AlignmentFlag.AlignLeft, f"{start_val:.0f}")
        painter.drawText(x_start + width - 50, y_start + height + 5, 50, 20, Qt.AlignmentFlag.AlignRight, f"{end_val:.0f}")

        # Axis titles
        painter.setPen(QColor(styles.Colors.ON_SURFACE))
        painter.drawText(x_start, y_start + height + 25, width, 20, Qt.AlignmentFlag.AlignCenter, "Blur Score (Variance)")

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
    try:
        if not os.path.exists(image_path):
            return (image_path, None)
        
        file_extension = os.path.splitext(image_path)[1].lower()
        rgb_image = None
        
        if file_extension in ['.cr3', '.dng', '.nef', '.arw']:
            with rawpy.imread(image_path) as raw:
                rgb_image = raw.postprocess(rawpy.Params(
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                    use_camera_wb=True,
                    no_auto_bright=True,
                    half_size=True
                ))
        else:
            rgb_image = cv2.imread(image_path)
            if rgb_image is None:
                return (image_path, None)
        
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        variance = laplacian.var()
        
        return (image_path, float(variance))
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}\n{traceback.format_exc()}")
        return (image_path, None)


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
    """Worker thread for image analysis."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, image_paths: List[str]):
        super().__init__()
        self.image_paths = image_paths
    
    def run(self):
        """Run analysis in background."""
        try:
            results = {}
            num_processes = cpu_count()
            total = len(self.image_paths)
            
            with Pool(processes=num_processes) as pool:
                it = pool.imap_unordered(process_image_worker, self.image_paths)
                for idx, (path, score) in enumerate(it):
                    results[path] = score
                    progress = int((idx + 1) / total * 100)
                    self.progress.emit(progress)
            
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
        title = QLabel("Blur Detector")
        title.setProperty("role", "heading")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)
        
        subtitle = QLabel("Advanced blur detection using Laplacian variance analysis")
        subtitle.setProperty("role", "subheading")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(subtitle)
        
        # Step Indicator (Moved to header)
        self.step_label = QLabel("Step 1 of 3")
        self.step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.step_label.setStyleSheet(f"color: {styles.Colors.PRIMARY}; font-weight: bold; letter-spacing: 1px; margin-top: 10px;")
        header_layout.addWidget(self.step_label)
        
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
        # self.stacked_widget.addWidget(self.create_step_results()) # Removed Step 4
        
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
        self.delete_btn = QPushButton("Delete Blurry")
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
        self.step_label.setText(f"Step {step + 1} of 3")
        
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
        
        self.path_label = QLabel("No paths selected")
        self.path_label.setStyleSheet(f"color: {styles.Colors.ERROR}; font-style: italic; margin-top: 10px;")
        layout.addWidget(self.path_label)
        
        # Gallery of selected items
        self.gallery_list = QListWidget()
        self.gallery_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery_list.setIconSize(QPixmap(100,100).size())
        self.gallery_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gallery_list.setMovement(QListWidget.Movement.Static)
        self.gallery_list.setSpacing(12)
        self.gallery_list.setWrapping(True)
        self.gallery_list.setMinimumHeight(250)
        layout.addWidget(self.gallery_list)
        
        return widget
    
    def create_step_analyze(self) -> QWidget:
        """Create Step 2: Analyze Images."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label = QLabel("Analyze Images")
        label.setProperty("role", "heading")
        layout.addWidget(label)
        
        desc = QLabel("Click 'Start Analysis' to process your images:")
        desc.setProperty("role", "subheading")
        layout.addWidget(desc)
        
        layout.addSpacing(20)
        
        btn_layout = QHBoxLayout()
        btn = QPushButton("Start Analysis")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(self.start_analysis)
        btn_layout.addWidget(btn)
        
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
        self.bin_info_label = QLabel("Click on a bar in the histogram to view images in that range.")
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
        
        desc = QLabel("Set a blur score threshold to identify blurry images:")
        desc.setProperty("role", "subheading")
        layout.addWidget(desc)
        
        layout.addSpacing(20)
        
        filter_layout = QHBoxLayout()
        threshold_label = QLabel("Threshold:")
        filter_layout.addWidget(threshold_label)
        
        self.threshold_input = QLineEdit("100.0")
        self.threshold_input.setMaximumWidth(100)
        filter_layout.addWidget(self.threshold_input)
        
        btn = QPushButton("Filter & Show Results")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(self.filter_results)
        filter_layout.addWidget(btn)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

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
    
    def create_step_results(self) -> QWidget:
        """Create Step 4: Results & Actions."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label = QLabel("Handle Blurry Images")
        label.setProperty("role", "heading")
        layout.addWidget(label)
        
        desc = QLabel("Choose an action for blurry images:")
        desc.setProperty("role", "subheading")
        layout.addWidget(desc)
        
        layout.addSpacing(20)
        
        action_layout = QHBoxLayout()
        
        delete_btn = QPushButton("Delete Blurry Images")
        delete_btn.setProperty("role", "danger")
        delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        delete_btn.clicked.connect(self.delete_blurry)
        action_layout.addWidget(delete_btn)
        
        move_btn = QPushButton("Move Blurry Images")
        move_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        move_btn.clicked.connect(self.move_blurry)
        action_layout.addWidget(move_btn)
        
        action_layout.addStretch()
        layout.addLayout(action_layout)
        
        info = QLabel("✓ You can now start over with new images or close the application.")
        info.setStyleSheet(f"color: {styles.Colors.SUCCESS}; margin-top: 30px; font-weight: bold;")
        layout.addWidget(info)
        
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
            self.path_label.setStyleSheet("color: #388e3c; font-weight: bold;")
        else:
            self.path_label.setText("No paths selected")
            self.path_label.setStyleSheet("color: #d32f2f; font-style: italic;")
        # also refresh gallery
        try:
            self.update_gallery()
        except Exception:
            pass
    
    def get_image_paths(self, path: str) -> List[str]:
        """Find all image files in a path."""
        image_files: List[str] = []
        IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg', '.cr3', '.dng', '.nef', '.arw']
        
        normalized_path = os.path.abspath(path)
        
        if not os.path.exists(normalized_path):
            return image_files
        
        if os.path.isfile(normalized_path):
            _, ext = os.path.splitext(normalized_path)
            if ext.lower() in IMAGE_EXTENSIONS:
                image_files.append(normalized_path)
            return image_files
        
        if os.path.isdir(normalized_path):
            for dirpath, _, filenames in os.walk(normalized_path):
                for filename in filenames:
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in IMAGE_EXTENSIONS:
                        full_path = os.path.join(dirpath, filename)
                        image_files.append(full_path)
        
        return image_files

    def update_filtered_list(self):
        """Update the QListWidget in Step 3 with filtered blurry images."""
        self.filtered_list.clear()
        for p in self.blurry_paths:
            item = QListWidgetItem(os.path.basename(p))
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.filtered_list.addItem(item)

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
                # use light placeholder until thumbnail is ready
                pm = QPixmap(128, 128)
                pm.fill(QColor('#eeeeee'))
                painter = QPainter(pm)
                painter.setPen(QPen(Qt.GlobalColor.black))
                painter.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, base[:15])
                painter.end()
                item.setIcon(QIcon(pm))
                files_to_thumb.append(p)
            else:
                # folder placeholder
                pm = QPixmap(128, 128)
                pm.fill(QColor('#f0f0f0'))
                painter = QPainter(pm)
                painter.setPen(QPen(Qt.GlobalColor.black))
                painter.drawRect(0, 0, 127, 127)
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
        """Open image in the system default file explorer, with the file selected."""
        import subprocess
        import platform
        
        if not os.path.exists(path):
            QMessageBox.critical(self, "Error", "Image not found on disk.")
            return
        
        try:
            system = platform.system()
            abs_path = os.path.abspath(path)
            
            if system == "Darwin":  # macOS
                # Use 'open -R' to reveal file in Finder
                subprocess.Popen(["open", "-R", abs_path])
            elif system == "Windows":
                # Use 'explorer /select,' to select file
                subprocess.Popen(f'explorer /select, "{abs_path}"')
            elif system == "Linux":
                # Try xdg-open with the folder, or use dbus if available
                folder = os.path.dirname(abs_path)
                subprocess.Popen(["xdg-open", folder])
            else:
                QMessageBox.warning(self, "Warning", f"Unsupported OS: {system}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open in explorer: {str(e)}")
    
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
            self.image_paths_to_process.extend(self.get_image_paths(path))
        
        self.image_paths_to_process = list(dict.fromkeys(self.image_paths_to_process))
        
        if not self.image_paths_to_process:
            QMessageBox.critical(self, "Error", "No images found in selected paths!")
            return
        
        # Start analysis thread
        self.status_label.setText("Analyzing...")
        self.status_label.setStyleSheet("color: #f57c00; font-weight: bold;")
        self.progress_bar.setValue(0)
        
        self.analysis_thread = AnalysisThread(self.image_paths_to_process)
        self.analysis_thread.progress.connect(self.update_progress)
        self.analysis_thread.finished.connect(self.on_analysis_complete)
        self.analysis_thread.error.connect(self.on_analysis_error)
        self.analysis_thread.start()
    
    def update_progress(self, value: int):
        """Update progress bar."""
        self.progress_bar.setValue(value)
    
    def on_analysis_complete(self, results: Dict):
        """Handle analysis completion."""
        self.results = results
        self.status_label.setText("✓ Analysis complete!")
        self.status_label.setStyleSheet(f"color: {styles.Colors.SUCCESS}; font-weight: bold;")
        self.histogram_widget.set_data(results)
        # self.display_results() # Removed as per request
    
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
        self.blurry_paths = [p for p in self.results if self.results[p] is not None and self.results[p] < threshold]

        # Populate the filtered list widget with filenames and store full path in item data
        self.update_filtered_list()

        if not self.blurry_paths:
            QMessageBox.information(self, "No Blurry Images", "No blurry images found with this threshold.")
    
    def delete_blurry(self):
        """Delete blurry images."""
        if not self.blurry_paths:
            QMessageBox.warning(self, "Warning", "Filter results first!")
            return
        
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Delete {len(self.blurry_paths)} blurry image(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                for path in self.blurry_paths:
                    os.remove(path)
                QMessageBox.information(self, "Success", f"Deleted {len(self.blurry_paths)} image(s)!")
                self.blurry_paths = []
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete: {str(e)}")
    
    def move_blurry(self):
        """Move blurry images."""
        if not self.blurry_paths:
            QMessageBox.warning(self, "Warning", "Filter results first!")
            return
        
        dest = QFileDialog.getExistingDirectory(self, "Select destination folder")
        if not dest:
            return
        
        try:
            if not os.path.exists(dest):
                os.makedirs(dest)
            
            for path in self.blurry_paths:
                filename = os.path.basename(path)
                shutil.move(path, os.path.join(dest, filename))
            
            QMessageBox.information(self, "Success", f"Moved {len(self.blurry_paths)} image(s)!")
            self.blurry_paths = []
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to move: {str(e)}")


class SelectionDialog(QDialog):
    """Dialog for selecting files and/or folders."""
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Select Images or Folders")
        self.setGeometry(200, 200, 400, 250)
        self.selected_paths: List[str] = []
        
        layout = QVBoxLayout()
        
        label = QLabel("What would you like to select?")
        label_font = QFont()
        label_font.setPointSize(11)
        label.setFont(label_font)
        layout.addWidget(label)
        
        self.button_group = QButtonGroup()
        
        rb1 = QRadioButton("📄 Image files")
        rb2 = QRadioButton("📁 Folder(s)")
        rb3 = QRadioButton("📄 Both (files and folders)")
        rb3.setChecked(True)
        
        self.button_group.addButton(rb1, 1)
        self.button_group.addButton(rb2, 2)
        self.button_group.addButton(rb3, 3)
        
        layout.addWidget(rb1)
        layout.addWidget(rb2)
        layout.addWidget(rb3)
        
        layout.addStretch()
        
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        
        ok_btn.clicked.connect(self.confirm)
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def confirm(self):
        """Confirm selection."""
        choice = self.button_group.checkedId()
        
        if choice == 1 or choice == 3:  # Files or both
            files = QFileDialog.getOpenFileNames(
                self, "Select image files",
                filter="Images (*.jpg *.jpeg *.png *.gif *.bmp *.tiff *.webp *.cr3 *.dng *.nef *.arw);;All Files (*)"
            )[0]
            self.selected_paths.extend(files)
        
        if choice == 2 or choice == 3:  # Folders or both
            folder = QFileDialog.getExistingDirectory(self, "Select folder containing images")
            if folder:
                self.selected_paths.append(folder)
        
        if self.selected_paths:
            self.accept()
        else:
            QMessageBox.warning(self, "Warning", "Please select at least one file or folder!")
    
    def get_selected_paths(self) -> List[str]:
        """Get selected paths."""
        return self.selected_paths


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BlurDetectorGUI()
    window.show()
    sys.exit(app.exec())
