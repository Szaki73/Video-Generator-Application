# draggable_label.py
from PySide6.QtWidgets import QLabel

class ClickableLabel(QLabel):
    def __init__(self, cam, controller, parent=None):
        super().__init__(parent)
        self.cam = cam
        self.controller = controller

    def mousePressEvent(self, event):
        self.controller.handle_click(self.cam)
