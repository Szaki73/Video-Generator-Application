from PySide6.QtWidgets import QSpinBox
from PySide6.QtGui import QValidator

class DefaultSpinBox(QSpinBox):
    def __init__(self, parent=None, default_label="default"):
        super().__init__(parent)
        self.default_label = default_label

    def textFromValue(self, value):
        if value == 0:
            return self.default_label
        return str(value)

    def valueFromText(self, text):
        if text == self.default_label:
            return 0
        try:
            return int(text)
        except ValueError:
            return 0

    def validate(self, text, pos):
        if text == self.default_label:
            return (QValidator.Acceptable, text, pos)
        return super().validate(text, pos)
