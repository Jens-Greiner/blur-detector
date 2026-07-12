
from PyQt6.QtGui import QColor, QPalette

# Material Design 3 Color Palette (light), tightened into one named set.
# The BLURRY/SHARP pair is the app's semantic split: everything below a chosen
# threshold reads as a "reject" (blurry), everything above as a "keeper" (sharp).
class Colors:
    PRIMARY = "#6750A4"       # M3 Purple
    PRIMARY_HOVER = "#7F67BE"
    PRIMARY_PRESSED = "#4F378B"
    ON_PRIMARY = "#FFFFFF"
    PRIMARY_CONTAINER = "#EADDFF"
    ON_PRIMARY_CONTAINER = "#21005D"

    SECONDARY = "#625B71"
    ON_SECONDARY = "#FFFFFF"
    SECONDARY_CONTAINER = "#E8DEF8"
    ON_SECONDARY_CONTAINER = "#1D192B"

    TERTIARY = "#7D5260"
    ON_TERTIARY = "#FFFFFF"
    TERTIARY_CONTAINER = "#FFD8E4"
    ON_TERTIARY_CONTAINER = "#31111D"

    ERROR = "#B3261E"
    ERROR_HOVER = "#C5372F"
    ON_ERROR = "#FFFFFF"

    BACKGROUND = "#FCFAFF"
    ON_BACKGROUND = "#1C1B1F"

    SURFACE = "#FFFFFF"
    ON_SURFACE = "#1C1B1F"

    # Subtle elevation tint for cards sitting on the background
    SURFACE_CONTAINER = "#F6F2FB"
    SURFACE_VARIANT = "#E7E0EC"
    ON_SURFACE_VARIANT = "#49454F"

    OUTLINE = "#79747E"
    OUTLINE_VARIANT = "#CAC4D0"

    # Semantic split — the signature of the tool
    BLURRY = "#D9603B"        # warm red-orange: reject / soft
    BLURRY_CONTAINER = "#FBE6DE"
    SHARP = "#2E9E8F"         # teal-green: keeper / sharp
    SHARP_CONTAINER = "#D8F0EC"

    # Status
    SUCCESS = "#2E9E8F"
    WARNING = "#FF9800"


# Font stacks. Scores are data, so they get a tabular monospace face.
FONT_SANS = '"Segoe UI", "Roboto", "Helvetica Neue", "SF Pro Text", sans-serif'
FONT_MONO = '"SF Mono", "JetBrains Mono", "Roboto Mono", "Menlo", "Consolas", monospace'


def get_stylesheet() -> str:
    return f"""
    QMainWindow {{
        background-color: {Colors.BACKGROUND};
    }}

    QWidget {{
        font-family: {FONT_SANS};
        font-size: 14px;
        color: {Colors.ON_BACKGROUND};
    }}

    /* ---- Type scale ---- */
    QLabel[role="display"] {{
        font-size: 30px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: {Colors.ON_BACKGROUND};
    }}

    QLabel[role="heading"] {{
        font-size: 22px;
        font-weight: 700;
        color: {Colors.ON_SURFACE};
    }}

    QLabel[role="subheading"] {{
        font-size: 15px;
        color: {Colors.ON_SURFACE_VARIANT};
    }}

    QLabel[role="caption"] {{
        font-size: 12px;
        color: {Colors.ON_SURFACE_VARIANT};
    }}

    QLabel[role="mono"] {{
        font-family: {FONT_MONO};
        font-size: 13px;
        color: {Colors.ON_SURFACE};
    }}

    /* ---- Cards ---- */
    QFrame[role="card"] {{
        background-color: {Colors.SURFACE};
        border: 1px solid {Colors.OUTLINE_VARIANT};
        border-radius: 16px;
    }}

    /* ---- Buttons ---- */
    QPushButton {{
        background-color: {Colors.PRIMARY};
        color: {Colors.ON_PRIMARY};
        border: none;
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 14px;
    }}

    QPushButton:hover {{
        background-color: {Colors.PRIMARY_HOVER};
    }}

    QPushButton:pressed {{
        background-color: {Colors.PRIMARY_PRESSED};
    }}

    QPushButton:disabled {{
        background-color: {Colors.SURFACE_VARIANT};
        color: {Colors.ON_SURFACE_VARIANT};
    }}

    /* Secondary Button (Outlined) */
    QPushButton[role="secondary"] {{
        background-color: transparent;
        border: 1px solid {Colors.OUTLINE};
        color: {Colors.PRIMARY};
    }}

    QPushButton[role="secondary"]:hover {{
        background-color: {Colors.PRIMARY_CONTAINER};
        border-color: {Colors.PRIMARY};
    }}

    QPushButton[role="secondary"]:pressed {{
        background-color: {Colors.SECONDARY_CONTAINER};
    }}

    /* Danger Button */
    QPushButton[role="danger"] {{
        background-color: {Colors.ERROR};
        color: {Colors.ON_ERROR};
    }}

    QPushButton[role="danger"]:hover {{
        background-color: {Colors.ERROR_HOVER};
    }}

    /* ---- Inputs ---- */
    QLineEdit, QSpinBox, QComboBox {{
        border: 1px solid {Colors.OUTLINE};
        border-radius: 8px;
        padding: 8px 10px;
        background-color: {Colors.SURFACE};
        selection-background-color: {Colors.PRIMARY_CONTAINER};
        selection-color: {Colors.ON_PRIMARY_CONTAINER};
    }}

    QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
        border: 2px solid {Colors.PRIMARY};
        padding: 7px 9px; /* compensate for the thicker border */
    }}

    /* ---- List Widget ---- */
    QListWidget {{
        background-color: {Colors.SURFACE};
        border: 1px solid {Colors.OUTLINE_VARIANT};
        border-radius: 12px;
        outline: none;
        padding: 4px;
    }}

    QListWidget::item {{
        padding: 8px;
        border-radius: 8px;
        margin: 2px;
    }}

    QListWidget::item:selected {{
        background-color: {Colors.PRIMARY_CONTAINER};
        color: {Colors.ON_PRIMARY_CONTAINER};
    }}

    QListWidget::item:hover {{
        background-color: {Colors.SURFACE_CONTAINER};
    }}

    /* ---- ProgressBar ---- */
    QProgressBar {{
        border: none;
        background-color: {Colors.SURFACE_VARIANT};
        border-radius: 5px;
        text-align: center;
        height: 10px;
    }}

    QProgressBar::chunk {{
        background-color: {Colors.PRIMARY};
        border-radius: 5px;
    }}

    /* ---- ScrollBar (minimal) ---- */
    QScrollBar:vertical {{
        border: none;
        background: transparent;
        width: 6px;
        margin: 0px;
    }}

    QScrollBar::handle:vertical {{
        background: rgba(0, 0, 0, 0.2);
        min-height: 20px;
        border-radius: 3px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: rgba(0, 0, 0, 0.4);
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    QScrollBar:horizontal {{
        border: none;
        background: transparent;
        height: 6px;
        margin: 0px;
    }}

    QScrollBar::handle:horizontal {{
        background: rgba(0, 0, 0, 0.2);
        min-width: 20px;
        border-radius: 3px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background: rgba(0, 0, 0, 0.4);
    }}

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    /* ---- Transparent containers ---- */
    QScrollArea {{
        background-color: transparent;
        border: none;
    }}

    QScrollArea > QWidget > QWidget {{
        background-color: transparent;
    }}

    QStackedWidget {{
        background-color: transparent;
    }}
    """
