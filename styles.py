
from PyQt6.QtGui import QColor, QPalette

# Material Design 3 Color Palette (approximate)
class Colors:
    PRIMARY = "#6750A4"       # M3 Purple
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
    ON_ERROR = "#FFFFFF"
    
    BACKGROUND = "#FFFBFE"
    ON_BACKGROUND = "#1C1B1F"
    
    SURFACE = "#FFFBFE"
    ON_SURFACE = "#1C1B1F"
    
    SURFACE_VARIANT = "#E7E0EC"
    ON_SURFACE_VARIANT = "#49454F"
    
    OUTLINE = "#79747E"

    # Custom additions
    SUCCESS = "#4CAF50"
    WARNING = "#FF9800"

def get_stylesheet() -> str:
    return f"""
    QMainWindow {{
        background-color: {Colors.BACKGROUND};
    }}
    
    QWidget {{
        font-family: "Segoe UI", "Roboto", "Helvetica Neue", sans-serif;
        font-size: 14px;
        color: {Colors.ON_BACKGROUND};
    }}
    
    /* Headings */
    QLabel[role="heading"] {{
        font-size: 24px;
        font-weight: bold;
        color: {Colors.PRIMARY};
    }}
    
    QLabel[role="subheading"] {{
        font-size: 16px;
        color: {Colors.ON_SURFACE_VARIANT};
    }}
    
    /* Cards */
    QFrame[role="card"] {{
        background-color: {Colors.SURFACE};
        border: 1px solid {Colors.SURFACE_VARIANT};
        border-radius: 12px;
    }}
    
    /* Buttons */
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
        background-color: #7F67BE; /* Slightly lighter */
    }}
    
    QPushButton:pressed {{
        background-color: {Colors.ON_PRIMARY_CONTAINER};
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
    
    /* Danger Button */
    QPushButton[role="danger"] {{
        background-color: {Colors.ERROR};
        color: {Colors.ON_ERROR};
    }}
    
    QPushButton[role="danger"]:hover {{
        background-color: #DC362E;
    }}
    
    /* Inputs */
    QLineEdit, QSpinBox, QComboBox {{
        border: 1px solid {Colors.OUTLINE};
        border-radius: 4px;
        padding: 8px;
        background-color: {Colors.SURFACE};
        selection-background-color: {Colors.PRIMARY_CONTAINER};
        selection-color: {Colors.ON_PRIMARY_CONTAINER};
    }}
    
    QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
        border: 2px solid {Colors.PRIMARY};
        padding: 7px; /* Adjust for border width change */
    }}
    
    /* List Widget */
    QListWidget {{
        background-color: {Colors.SURFACE};
        border: 1px solid {Colors.SURFACE_VARIANT};
        border-radius: 8px;
        outline: none;
    }}
    
    QListWidget::item {{
        padding: 8px;
        border-radius: 4px;
        margin: 2px;
    }}
    
    QListWidget::item:selected {{
        background-color: {Colors.PRIMARY_CONTAINER};
        color: {Colors.ON_PRIMARY_CONTAINER};
    }}
    
    QListWidget::item:hover {{
        background-color: {Colors.SURFACE_VARIANT};
    }}
    
    /* ProgressBar */
    QProgressBar {{
        border: none;
        background-color: {Colors.SURFACE_VARIANT};
        border-radius: 4px;
        text-align: center;
        height: 8px;
    }}
    
    QProgressBar::chunk {{
        background-color: {Colors.PRIMARY};
        border-radius: 4px;
    }}
    
    /* ScrollBar */
    QScrollBar:vertical {{
        border: none;
        background: {Colors.SURFACE_VARIANT};
        width: 10px;
        border-radius: 5px;
    }}
    
    QScrollBar::handle:vertical {{
        background: {Colors.OUTLINE};
        min-height: 20px;
        border-radius: 5px;
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    /* ScrollArea and StackedWidget Backgrounds */
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
