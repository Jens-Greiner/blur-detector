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
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QIcon, QPixmap, QPainter, QPen, QBrush, QImage


class HistogramWidget(QWidget):
    """Custom histogram widget for displaying blur score distribution."""
    def __init__(self):
        super().__init__()
        self.scores = []
        self.setMinimumHeight(200)
        self.setStyleSheet("background-color: white; border: 1px solid #ddd; border-radius: 5px;")
    
    def set_data(self, scores: List[float]):
        """Update histogram with new scores."""
        self.scores = scores
        self.update()
    
    def paintEvent(self, event):
        """Draw the histogram."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self.scores:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No data available")
            return

        # Calculate histogram
        bins = min(20, len(self.scores))
        lo = float(min(self.scores))
        hi = float(max(self.scores))

        if lo == hi:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"All scores equal: {lo:.2f}")
            return

        counts = [0] * bins
        for s in self.scores:
            bin_idx = int((s - lo) / (hi - lo) * (bins - 0.0001))
            counts[bin_idx] += 1

        # Draw histogram area
        left_margin = 60
        right_margin = 20
        top_margin = 20
        bottom_margin = 50

        width = self.width() - left_margin - right_margin
        height = self.height() - top_margin - bottom_margin
        x_start = left_margin
        y_start = top_margin

        max_count = max(counts) if counts else 1
        bar_width = width / bins if bins > 0 else width

        painter.setPen(QPen(Qt.GlobalColor.black, 1))
        # axes
        painter.drawLine(x_start, y_start + height, x_start + width, y_start + height)  # x axis
        painter.drawLine(x_start, y_start, x_start, y_start + height)  # y axis

        # bars
        for i, count in enumerate(counts):
            bar_height = (count / max_count) * height if max_count > 0 else 0
            x = x_start + i * bar_width
            y = y_start + height - bar_height
            painter.fillRect(int(x), int(y), max(1, int(bar_width - 1)), int(bar_height), QBrush(QColor("#1976d2")))
            painter.drawRect(int(x), int(y), max(1, int(bar_width - 1)), int(bar_height))

        # Draw y-axis ticks and labels (choose ~5 ticks)
        painter.setPen(QPen(Qt.GlobalColor.black))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)

        n_yticks = 5
        for t in range(n_yticks + 1):
            val = int(round(t * (max_count / n_yticks)))
            y = y_start + height - (t / n_yticks) * height
            painter.drawLine(x_start - 5, int(y), x_start, int(y))
            painter.drawText(6, int(y + 4), f"{val}")

        # Draw x-axis tick labels: use up to 10 labels to avoid overlap
        max_labels = 10
        step = max(1, bins // max_labels)
        for i in range(0, bins, step):
            bin_lo = lo + i * (hi - lo) / bins
            bin_hi = lo + (i + 1) * (hi - lo) / bins
            label_val = (bin_lo + bin_hi) / 2.0
            x = x_start + i * bar_width + bar_width / 2
            txt = f"{label_val:.1f}"
            painter.drawLine(int(x), y_start + height, int(x), y_start + height + 5)
            painter.drawText(int(x - 15), y_start + height + 20, txt)

        # Axis titles
        painter.drawText(x_start + width // 2 - 40, y_start + height + 40, "Blur Score")
        painter.drawText(10, y_start + height // 2, "Freq")


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
        self.apply_stylesheet()
        self.show_step(0)
    
    def setup_ui(self):
        """Create the main UI with stacked pages."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Blur Detector")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        subtitle = QLabel("Advanced blur detection using Laplacian variance analysis")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #666; font-size: 11pt;")
        main_layout.addWidget(subtitle)
        
        main_layout.addSpacing(10)
        
        # Stacked widget for steps
        self.stacked_widget = QStackedWidget()
        
        # Step 0: Select Images
        self.stacked_widget.addWidget(self.create_step_select_images())
        
        # Step 1: Analyze Images
        self.stacked_widget.addWidget(self.create_step_analyze())
        
        # Step 2: Filter Results
        self.stacked_widget.addWidget(self.create_step_filter())
        
        # Step 3: Results & Actions
        self.stacked_widget.addWidget(self.create_step_results())
        
        main_layout.addWidget(self.stacked_widget)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("← Back")
        self.prev_btn.setMinimumHeight(40)
        self.prev_btn.setMinimumWidth(100)
        self.prev_btn.setStyleSheet("""
            QPushButton {
                background-color: #bbb;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #999; }
        """)
        self.prev_btn.clicked.connect(self.prev_step)
        nav_layout.addWidget(self.prev_btn)
        
        nav_layout.addStretch()
        
        self.step_label = QLabel("Step 1 of 4")
        self.step_label.setStyleSheet("font-weight: bold; color: #1976d2;")
        nav_layout.addWidget(self.step_label)
        
        nav_layout.addStretch()
        
        self.next_btn = QPushButton("Next →")
        self.next_btn.setMinimumHeight(40)
        self.next_btn.setMinimumWidth(100)
        self.next_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1565c0; }
        """)
        self.next_btn.clicked.connect(self.next_step)
        nav_layout.addWidget(self.next_btn)
        
        main_layout.addLayout(nav_layout)
    
    def show_step(self, step: int):
        """Show a specific step."""
        self.current_step = step
        self.stacked_widget.setCurrentIndex(step)
        self.step_label.setText(f"Step {step + 1} of 4")
        
        # Update button visibility
        self.prev_btn.setEnabled(step > 0)
        
        if step == 0:
            self.next_btn.setText("Next →")
        elif step == 3:
            self.next_btn.setText("Finish")
        else:
            self.next_btn.setText("Next →")
    
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
        elif self.current_step == 3:  # Results
            QMessageBox.information(self, "Complete", "Blur detection workflow complete!")
            return
        
        if self.current_step < 3:
            self.show_step(self.current_step + 1)
    
    def create_step_select_images(self) -> QWidget:
        """Create Step 1: Select Images."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label = QLabel("Step 1: Select Images or Folders")
        label_font = QFont()
        label_font.setPointSize(14)
        label_font.setBold(True)
        label.setFont(label_font)
        label.setStyleSheet("color: #1976d2;")
        layout.addWidget(label)
        
        desc = QLabel("Choose image files or folders to analyze:")
        desc.setStyleSheet("color: #666; margin-bottom: 20px;")
        layout.addWidget(desc)
        
        btn_layout = QHBoxLayout()
        btn_files = QPushButton("🖼️ Add Images")
        btn_files.setMinimumHeight(40)
        btn_files.setStyleSheet("""
            QPushButton {
                background-color: #0288d1;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0277bd; }
        """)
        btn_files.clicked.connect(self.add_image_files)
        btn_layout.addWidget(btn_files)
        
        btn_folder = QPushButton("📁 Add Folder")
        btn_folder.setMinimumHeight(40)
        btn_folder.setStyleSheet("""
            QPushButton {
                background-color: #0288d1;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0277bd; }
        """)
        btn_folder.clicked.connect(self.add_folder)
        btn_layout.addWidget(btn_folder)
        
        btn_clear = QPushButton("Clear")
        btn_clear.setMinimumHeight(40)
        btn_clear.setStyleSheet("""
            QPushButton { background-color: #eee; border: 1px solid #ddd; border-radius: 5px; color: black; }
            QPushButton:hover { background-color: #e0e0e0; }
        """)
        btn_clear.clicked.connect(self.clear_selection)
        btn_layout.addWidget(btn_clear)
        
        layout.addLayout(btn_layout)
        
        self.path_label = QLabel("No paths selected")
        self.path_label.setStyleSheet("color: #d32f2f; font-style: italic; margin-top: 20px; padding: 15px; background-color: #f5f5f5; border-radius: 5px;")
        layout.addWidget(self.path_label)
        
        # Gallery of selected items
        self.gallery_list = QListWidget()
        self.gallery_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery_list.setIconSize(QPixmap(128,128).size())
        self.gallery_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gallery_list.setMovement(QListWidget.Movement.Static)
        self.gallery_list.setSpacing(10)
        self.gallery_list.setWrapping(True)
        self.gallery_list.setMinimumHeight(200)
        layout.addWidget(self.gallery_list)
        
        layout.addStretch()
        return widget
    
    def create_step_analyze(self) -> QWidget:
        """Create Step 2: Analyze Images."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label = QLabel("Step 2: Analyze Images")
        label_font = QFont()
        label_font.setPointSize(14)
        label_font.setBold(True)
        label.setFont(label_font)
        label.setStyleSheet("color: #1976d2;")
        layout.addWidget(label)
        
        desc = QLabel("Click 'Start Analysis' to process your images:")
        desc.setStyleSheet("color: #666; margin-bottom: 20px;")
        layout.addWidget(desc)
        
        btn_layout = QHBoxLayout()
        btn = QPushButton("▶ Start Analysis")
        btn.setMinimumHeight(40)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #388e3c;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2e7d32; }
        """)
        btn.clicked.connect(self.start_analysis)
        btn_layout.addWidget(btn)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #388e3c; font-weight: bold; margin-left: 20px;")
        btn_layout.addWidget(self.status_label)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setMinimumHeight(30)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #1976d2;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Histogram widget
        self.histogram_widget = HistogramWidget()
        layout.addWidget(self.histogram_widget)
        
        # Sample results label (shows up to 10 evenly spread samples)
        self.sample_label = QLabel("")
        self.sample_label.setStyleSheet("font-family: Courier; background-color: #fafafa; border: 1px solid #eee; padding: 8px;")
        self.sample_label.setMinimumHeight(120)
        self.sample_label.setWordWrap(True)
        layout.addWidget(self.sample_label)
        
        layout.addStretch()
        return widget
    
    def create_step_filter(self) -> QWidget:
        """Create Step 3: Filter Results."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label = QLabel("Step 3: Filter Results")
        label_font = QFont()
        label_font.setPointSize(14)
        label_font.setBold(True)
        label.setFont(label_font)
        label.setStyleSheet("color: #1976d2;")
        layout.addWidget(label)
        
        desc = QLabel("Set a blur score threshold to identify blurry images:")
        desc.setStyleSheet("color: #666; margin-bottom: 20px;")
        layout.addWidget(desc)
        
        filter_layout = QHBoxLayout()
        threshold_label = QLabel("Threshold:")
        threshold_label.setStyleSheet("color: black;")
        filter_layout.addWidget(threshold_label)
        
        self.threshold_input = QLineEdit("100.0")
        self.threshold_input.setMaximumWidth(100)
        self.threshold_input.setStyleSheet("QLineEdit { color: black; }")
        filter_layout.addWidget(self.threshold_input)
        
        btn = QPushButton("🔍 Filter & Show Results")
        btn.setMinimumHeight(40)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #f57c00;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #e65100; }
        """)
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
        self.open_btn.setMinimumHeight(36)
        self.open_btn.setStyleSheet("""
            QPushButton { background-color: #1976d2; color: white; border-radius: 5px; }
            QPushButton:hover { background-color: #1565c0; }
        """)
        self.open_btn.clicked.connect(self.open_selected_image)
        open_layout.addWidget(self.open_btn)
        open_layout.addStretch()
        layout.addLayout(open_layout)
        
        layout.addStretch()
        return widget
    
    def create_step_results(self) -> QWidget:
        """Create Step 4: Results & Actions."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label = QLabel("Step 4: Handle Blurry Images")
        label_font = QFont()
        label_font.setPointSize(14)
        label_font.setBold(True)
        label.setFont(label_font)
        label.setStyleSheet("color: #1976d2;")
        layout.addWidget(label)
        
        desc = QLabel("Choose an action for blurry images:")
        desc.setStyleSheet("color: #666; margin-bottom: 20px;")
        layout.addWidget(desc)
        
        action_layout = QHBoxLayout()
        
        delete_btn = QPushButton("🗑️ Delete Blurry Images")
        delete_btn.setMinimumHeight(40)
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #c62828; }
        """)
        delete_btn.clicked.connect(self.delete_blurry)
        action_layout.addWidget(delete_btn)
        
        move_btn = QPushButton("📦 Move Blurry Images")
        move_btn.setMinimumHeight(40)
        move_btn.setStyleSheet("""
            QPushButton {
                background-color: #7b1fa2;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #6a1b9a; }
        """)
        move_btn.clicked.connect(self.move_blurry)
        action_layout.addWidget(move_btn)
        
        action_layout.addStretch()
        layout.addLayout(action_layout)
        
        info = QLabel("✓ You can now start over with new images or close the application.")
        info.setStyleSheet("color: #388e3c; margin-top: 30px; font-weight: bold;")
        layout.addWidget(info)
        
        layout.addStretch()
        return widget
    
    def apply_stylesheet(self):
        """Apply modern stylesheet."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QLabel {
                color: #333;
            }
            QLineEdit {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
                background-color: white;
                selection-background-color: #1976d2;
            }
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
            }
        """)

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
        self.status_label.setStyleSheet("color: #388e3c; font-weight: bold;")
        self.draw_histogram()
        self.display_results()
    
    def draw_histogram(self):
        """Draw blur score histogram in the analysis section."""
        scores = [s for s in self.results.values() if s is not None]
        self.histogram_widget.set_data(scores)
    
    def on_analysis_error(self, error: str):
        """Handle analysis error."""
        self.status_label.setText("✗ Error")
        self.status_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
        QMessageBox.critical(self, "Analysis Error", error)
    
    def display_results(self):
        """Display analysis results."""
        sorted_results = sorted(
            [(p, s) for p, s in self.results.items() if s is not None], key=lambda x: x[1]
        )
        
        text = ""
        
        if not sorted_results:
            text = "No valid blur scores found."
        else:
            text = f"Found {len(sorted_results)} valid images\n\n"
            text += "Sample Results (evenly spread):\n\n"
            
            L = len(sorted_results)
            n_show = min(10, L)
            
            if n_show == 1:
                indices = [0]
            else:
                indices = [round(i * (L - 1) / (n_show - 1)) for i in range(n_show)]
            
            indices = sorted(set(indices))
            
            for idx in indices:
                path, score = sorted_results[idx]
                text += f"• {os.path.basename(path):40s} | Score: {score:8.2f}\n"
        
        # display into sample_label (analysis step) or print to terminal if missing
        if hasattr(self, 'sample_label') and self.sample_label is not None:
            self.sample_label.setText(text)
        else:
            print(text)
    
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
