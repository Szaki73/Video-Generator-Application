from PySide6.QtWidgets import QApplication
from generator_window import GeneratorWindow
import sys

def main():
    app = QApplication(sys.argv)
    window = GeneratorWindow()
    window.setWindowTitle("Video Generator")

    screen = app.primaryScreen()
    available_geometry = screen.availableGeometry()

    window.setMinimumSize(available_geometry.width() * 0.8, available_geometry.height() * 0.8)

    window.resize(available_geometry.width(), available_geometry.height())
    window.move(available_geometry.topLeft())

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# C:\Users\tamas\Documents\szakgyak\code\VideoGenerator\pictures2Div
# C:\Users\tamas\Documents\szakgyak\code\VideoGenerator\img2