from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QSpinBox,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
    QScrollArea, QStackedLayout
)
from PySide6 import QtWidgets
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
import sys
import os
from clickable_label import ClickableLabel
from default_spinbox import DefaultSpinBox
import cv2 as cv
import argparse
import sys
import os
import cv2 as cv
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
import time
import logging

if getattr(sys, 'frozen', False):
    # Running as a bundled executable
    base_path = os.path.dirname(sys.executable)
else:
    # Running as a script
    base_path = os.path.dirname(os.path.abspath(__file__))

log_file = os.path.join(base_path, "error.txt")

logging.basicConfig(
    filename=log_file,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)

class GeneratorWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.lidar_path = ""
        self.input_path = ""
        self.inputs = []
        self.output_path = os.path.join(base_path, "output")
        self.output_name = "output"
        self.row = 3
        self.column = 3
        self.columns = [3, 3, 3]
        self.framerate = 15
        self.start_frame = 0
        self.end_frame = 0
        self.camera_order = []
        self.camera_frames = {}
        self.frame_numbers = set()
        self.camera_delay_values = {}
        self.delay_vars = {}
        self.stop_generator = False
        self.errors = False

        self.camera_grid = []
        self.frame_data = {}
        self.image_width = None
        self.image_height = None
        self.video_width = None
        self.video_height = None
        self.image_labels = {}
        self.selected_cam = None 
        self.black_frame = None
        self.scale_factor = 1.0

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        setup_panel = QVBoxLayout()
        setup_panel.setAlignment(Qt.AlignTop)

        self.input_path_entry = QLineEdit()
        browse_input_btn = QPushButton("Browse")
        browse_input_btn.clicked.connect(self.browse_input)

        self.output_path_entry = QLineEdit(self.output_path)
        browse_output_btn = QPushButton("Browse")
        browse_output_btn.clicked.connect(self.browse_output)

        self.output_name_entry = QLineEdit(self.output_name)

        self.lidar_path_entry = QLineEdit()
        browse_lidar_input_btn = QPushButton("Browse")
        browse_lidar_input_btn.clicked.connect(self.browse_lidar_input)

        continue_btn = QPushButton("Continue")
        continue_btn.clicked.connect(self.show_grid_view)

        stop_btn = QPushButton("Stop Generator")
        stop_btn.clicked.connect(self.stop_gen)

        self.row_spin = QSpinBox()
        self.row_spin.setRange(1, 3)
        self.row_spin.setValue(self.row)
        self.row_spin.valueChanged.connect(self.update_row)

        self.col1_spin = QSpinBox()
        self.col1_spin.setRange(1, 3)
        self.col1_spin.setValue(self.columns[0])
        self.col1_spin.valueChanged.connect(lambda val: self.update_column(0, val))

        self.col2_spin = QSpinBox()
        self.col2_spin.setRange(1, 3)
        self.col2_spin.setValue(self.columns[1])
        self.col2_spin.valueChanged.connect(lambda val: self.update_column(1, val))

        self.col3_spin = QSpinBox()
        self.col3_spin.setRange(1, 3)
        self.col3_spin.setValue(self.columns[2])
        self.col3_spin.valueChanged.connect(lambda val: self.update_column(2, val))

        setup_panel.addWidget(QLabel("Input Path (required):"))
        setup_panel.addWidget(self.input_path_entry)
        setup_panel.addWidget(browse_input_btn)
        setup_panel.addWidget(QLabel("Lidar Input Path"))
        setup_panel.addWidget(self.lidar_path_entry)
        setup_panel.addWidget(browse_lidar_input_btn)
        setup_panel.addWidget(QLabel("Output Path:"))
        setup_panel.addWidget(self.output_path_entry)
        setup_panel.addWidget(browse_output_btn)
        setup_panel.addWidget(QLabel("Output Name:"))
        setup_panel.addWidget(self.output_name_entry)
        setup_panel.addWidget(continue_btn)
        setup_panel.addWidget(stop_btn)
        setup_panel.addWidget(QLabel("Row count:"))
        setup_panel.addWidget(self.row_spin)
        setup_panel.addWidget(QLabel("Columns in first row:"))
        setup_panel.addWidget(self.col1_spin)
        setup_panel.addWidget(QLabel("Columns in second row:"))
        setup_panel.addWidget(self.col2_spin)
        setup_panel.addWidget(QLabel("Columns in third row:"))
        setup_panel.addWidget(self.col3_spin)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red; font-weight: bold;")
        self.error_label.setWordWrap(True)
        setup_panel.addWidget(self.error_label)

        setup_frame = QFrame()
        setup_frame.setLayout(setup_panel)
        setup_frame.setFixedWidth(250)
        main_layout.addWidget(setup_frame)

        self.stacked_layout = QStackedLayout()
        self.grid_view = self.create_grid_view()
        placeholder = QWidget()
        self.stacked_layout.addWidget(placeholder)
        self.stacked_layout.addWidget(self.grid_view)

        stacked_frame = QFrame()
        stacked_frame.setLayout(self.stacked_layout)
        main_layout.addWidget(stacked_frame)

    def browse_input(self):
        path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if path:
            self.input_path_entry.setText(path)

    def browse_lidar_input(self):
        path = QFileDialog.getExistingDirectory(self, "Select Lidar Input Directory")
        if path:
            self.lidar_path_entry.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_path_entry.setText(path)

    def create_grid_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        input_bar = QGridLayout()
        self.framerate_spin = QSpinBox()
        self.framerate_spin.setRange(1, 120)
        self.framerate_spin.setValue(self.framerate)
        self.framerate_spin.valueChanged.connect(lambda val: setattr(self, 'framerate', val))

        self.start_frame_spin = DefaultSpinBox(default_label="The first")
        self.start_frame_spin.setRange(0, 9999)
        self.start_frame_spin.setValue(self.start_frame)
        self.start_frame_spin.valueChanged.connect(lambda val: setattr(self, 'start_frame', val))

        self.end_frame_spin = DefaultSpinBox(default_label="The last")
        self.end_frame_spin.setRange(0, 9999)
        self.end_frame_spin.setValue(self.end_frame)
        self.end_frame_spin.valueChanged.connect(lambda val: setattr(self, 'end_frame', val))

        self.global_delay_spin = QSpinBox()
        self.global_delay_spin.setRange(-9999, 9999)
        self.global_delay_spin.setValue(0)
        self.global_delay_spin.valueChanged.connect(self.on_global_delay_change)

        generate_btn = QPushButton("Generate")
        generate_btn.clicked.connect(self.generate_video)

        input_bar.addWidget(QLabel("Framerate:"), 0, 0)
        input_bar.addWidget(self.framerate_spin, 0, 1)
        input_bar.addWidget(QLabel("Start Frame:"), 0, 2)
        input_bar.addWidget(self.start_frame_spin, 0, 3)
        input_bar.addWidget(QLabel("End Frame:"), 0, 4)
        input_bar.addWidget(self.end_frame_spin, 0, 5)
        input_bar.addWidget(QLabel("Frame:"), 0, 6)
        input_bar.addWidget(self.global_delay_spin, 0, 7)
        input_bar.addWidget(generate_btn, 0, 8)

        layout.addLayout(input_bar)

        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_scroll.setWidget(self.grid_container)
        layout.addWidget(self.grid_scroll)

        self.image_width = self.grid_scroll.viewport().width()
        self.image_height = self.grid_scroll.viewport().height()

        return widget

    def update_row(self, value):
        self.row = value
        self.update_grid()

    def update_column(self, index, value):
        self.column = value
        self.columns[index] = value
        self.update_grid()

    def show_grid_view(self):
        self.inputs.clear()
        input_path = self.input_path_entry.text().strip()
        lidar_path = self.lidar_path_entry.text().strip()
        output_path = self.output_path_entry.text().strip()
        output_name = self.output_name_entry.text().strip()

        self.error_label.setText("")

        if not input_path:
            self.error_label.setText("Input path is required.")
            self.input_path_entry.setFocus()
            return

        if not os.path.isdir(input_path):
            self.error_label.setText(f"Input path does not exist")
            self.input_path_entry.setFocus()
            return

        if lidar_path and not os.path.isdir(lidar_path):
            self.error_label.setText(f"Lidar path does not exist")
            self.lidar_path_entry.setFocus()
            return

        if not os.path.isdir(output_path):
            self.error_label.setText(f"Output path does not exist")
            self.output_path_entry.setFocus()
            return

        self.input_path = input_path
        self.lidar_path = lidar_path
        self.output_path = output_path

        self.inputs.append(self.input_path)
        if self.lidar_path:
            self.inputs.append(self.lidar_path)

        self.output_name = output_name

        self.stacked_layout.setCurrentWidget(self.grid_view)
        self.load_first_frames()
        self.get_image_sizes()
        self.update_grid()

    def stop_gen(self):
        self.stop_generator = True

    def update_grid(self):
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        available_width = self.grid_scroll.viewport().width() * 0.9
        cell_width = available_width // 3
        cell_height = int(cell_width / self.image_width * self.image_height)

        self.camera_grid = []
        cam_index = 0

        for r in range(self.row):
            for c in range(self.columns[r]):
                cell_widget = QWidget()
                cell_layout = QVBoxLayout(cell_widget)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setSpacing(2)

                if cam_index < len(self.camera_order):
                    cam = self.camera_order[cam_index]
                    self.camera_grid.append(cam)

                    label = ClickableLabel(cam, controller=self)
                    label.setPixmap(self.get_frame_image(cam, frame_index=1))
                    label.setScaledContents(True)
                    label.setFixedSize(cell_width, cell_height - 30)
                    label.setAlignment(Qt.AlignCenter)
                    self.image_labels[cam] = label
                    cell_layout.addWidget(label)

                    delay_spin = QSpinBox()
                    delay_spin.setRange(-9999, 9999)
                    delay_spin.setValue(self.camera_delay_values.get(cam, 0))
                    delay_spin.valueChanged.connect(lambda val, c=cam: self.on_delay_change(c, val))
                    delay_spin.setFixedWidth(cell_width)
                    self.delay_vars[cam] = delay_spin
                    cell_layout.addWidget(delay_spin)

                    cam_index += 1
                else:
                    placeholder = QLabel("Empty")
                    placeholder.setAlignment(Qt.AlignCenter)
                    placeholder.setFixedSize(cell_width, cell_height)
                    cell_layout.addWidget(placeholder)

                self.grid_layout.addWidget(cell_widget, r, c)

    def load_first_frames(self):
        self.frame_data.clear()
        self.camera_order.clear()
        for inputs in self.inputs:
            image_files = sorted([f for f in os.listdir(inputs)])
            for f in image_files:
                parts = f.split("_")
                cam = parts[0]
                fn_part = [p for p in parts if p.startswith("fn")]
                if fn_part:
                    fn_str = fn_part[0][2:].split(".")[0]
                    fn = int(fn_str)
                    self.frame_data.setdefault(cam, {})[fn] = os.path.join(inputs, f)

        self.camera_order = list(self.frame_data.keys())
        self.camera_delay_values = {cam: 0 for cam in self.camera_order}

    def get_image_sizes(self):
        for cam in self.camera_order:
                frames = self.frame_data.get(cam, {})
                if frames:
                    first_path = frames.get(1) or next(iter(frames.values()))
                    img = cv.imread(first_path)
                    self.image_width = img.shape[1]
                    self.image_height =  img.shape[0]
                    return
    
    def get_frame_image(self, cam, frame_index):
        try:
            if cam not in self.frame_data:
                return self.get_black_image()

            delay = self.camera_delay_values.get(cam, 0)
            img_path = self.frame_data.get(cam, {}).get(frame_index + delay)
            if img_path is None:
                img_path = self.frame_data.get(cam, {}).get(0)
            if img_path is None:
                return self.get_black_image()

            img = cv.imread(img_path)
            if img is None:
                return self.get_black_image()

            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            qimg = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(self.image_width, self.image_height)
            return pixmap
        except Exception as e:
            return self.get_black_image()

    def get_black_image(self):
        black = QPixmap(self.image_width, self.image_height)
        black.fill("black")
        return black
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_grid()

    def handle_click(self, cam):
        if self.selected_cam is None:
            self.selected_cam = cam
        elif self.selected_cam == cam:
            self.selected_cam = None
        else:

            i1 = self.camera_grid.index(self.selected_cam)
            i2 = self.camera_grid.index(cam)
            self.camera_grid[i1], self.camera_grid[i2] = self.camera_grid[i2], self.camera_grid[i1]

            o1 = self.camera_order.index(self.selected_cam)
            o2 = self.camera_order.index(cam)
            self.camera_order[o1], self.camera_order[o2] = self.camera_order[o2], self.camera_order[o1]

            self.selected_cam = None
            self.update_grid()

    def update_frame(self, cam):
        frame_index = self.delay_vars.get(cam).value() + 1
        pixmap = self.get_frame_image(cam, frame_index)
        label = self.image_labels.get(cam)
        if label and pixmap:
            label.setPixmap(pixmap)

    def on_delay_change(self, cam, delay_val):
        self.camera_delay_values[cam] = delay_val
        self.update_frame(cam)

    def on_global_delay_change(self, new_global_delay):
        old_global_delay = getattr(self, "_last_global_delay", 0)
        delta = new_global_delay - old_global_delay

        for cam, spinbox in self.delay_vars.items():
            spinbox.setValue(spinbox.value() + delta)

        self._last_global_delay = new_global_delay

    def generate_video(self):
        with open("error.txt", "w") as f:
            f.write("")
        if self.stop_generator == True:
            self.stop_generator = False
        if self.errors == True:
            self.errors = False
        self.camera_order = self.camera_order[:self.row * self.column]
        if self.start_frame != 0 and self.end_frame != 0 and self.start_frame >= self.end_frame:
            self.error_label.setText(f"Starting frame cannot be greater or equal to ending frame!")
            return

        self.sort_images()
        self.get_video_height_and_video_width()

        self.black_frame =  np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        output_file = os.path.join(self.output_path, self.output_name + ".mp4")
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        video = cv.VideoWriter(output_file, fourcc, self.framerate, (self.video_width, self.video_height))

        self.frame_numbers = sorted(self.frame_numbers)

        if self.start_frame != 0:
            self.frame_numbers = [fn for fn in self.frame_numbers if fn >= self.start_frame]

        if self.end_frame != 0:
            self.frame_numbers = [fn for fn in self.frame_numbers if fn <= self.end_frame]

        self.camera_order = self.camera_order[:self.row * self.column]

        for index, fn in enumerate(self.frame_numbers, start=1):
            if self.stop_generator:
                video.release()
                self.stop_generator = False
                self.error_label.setText("Generation stopped")
                break

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.load_and_set_frame, cam, fn,
                                    self.camera_frames, self.image_height, self.image_width,
                                    self.black_frame, self.camera_order, self.camera_delay_values)
                    for cam in self.camera_order
                ]
                all_images = []
                for f in futures:
                    try:
                        result = f.result()
                        all_images.append(result)
                    except Exception as e:
                        logging.error(f"video_generator.py: error loading frame {fn} in thread: {e}")
                        self.errors = True
                        break

            canvas_height = self.video_height
            canvas_width = self.video_width

            scaled_image_width = int(self.image_width * self.scale_factor)
            scaled_image_height = int(self.image_height * self.scale_factor)

            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            image_index = 0
            for row_index in range(self.row):
                row_columns = self.columns[row_index]
                current_row = all_images[image_index:image_index + row_columns]
                image_index += row_columns

                if not current_row:
                    break

                vertical_offset = row_index * scaled_image_height
                total_row_width = len(current_row) * scaled_image_width
                horizontal_start = max(0, (canvas_width - total_row_width) // 2)

                for col_index, img in enumerate(current_row):
                    horizontal_offset = horizontal_start + col_index * scaled_image_width
                    resized = cv.resize(img, (scaled_image_width, scaled_image_height), interpolation=cv.INTER_AREA)
                    canvas[vertical_offset:vertical_offset + scaled_image_height,
                        horizontal_offset:horizontal_offset + scaled_image_width] = resized

            # ✅ Write the full canvas after all rows are drawn
            video.write(canvas)

            percent_complete = (index / len(self.frame_numbers)) * 100
            self.error_label.setText(f"Progress: {percent_complete:.2f}% ({index}/{len(self.frame_numbers)})")
            QtWidgets.QApplication.processEvents()

        video.release()
        if self.errors:
            self.error_label.setText("Video finished. Errors in the errors.txt.")
        else:
            self.error_label.setText("Video finished. No errors.")

    def sort_images(self):
        self.camera_order = self.camera_order[:self.row * sum(self.columns[:self.row])]
        self.camera_frames = {}
        self.frame_numbers = set()
        for inputs in self.inputs:
            image_files = [f for f in os.listdir(inputs)]

            for f in image_files:
                parts = f.split("_")
                cam = parts[0]
                if cam not in self.camera_order: continue
                cam_index = self.camera_order.index(cam)
                delay = int(self.camera_delay_values[self.camera_order[cam_index]])
                fn = int(parts[-1][2:].split(".")[0])
                self.camera_frames.setdefault(cam, {})[fn - delay] = os.path.join(inputs, f)
                self.frame_numbers.add(fn - delay)

    def get_video_height_and_video_width(self):
        active_cameras = len(self.camera_order)

        self.video_width = min(active_cameras, max(self.columns[:self.row])) * self.image_width
        video_rows = math.ceil(active_cameras / self.column)
        total = 0
        for i, count in enumerate(self.columns):
            total += count
            if total >= active_cameras:
                video_rows = i + 1
                break
        self.video_height = video_rows * self.image_height
        if self.video_width > 4096:
            self.scale_factor = 4096 / self.video_width
            self.video_width = 4096
            self.video_height = video_rows * int(self.image_height * self.scale_factor)
        else:
            self.scale_factor = 1.0


    def load_and_set_frame(self, cam, fn, camera_frames, height, width, black_frame, camera_order, camera_delay_values):
        try:
            img_file = self.camera_frames[cam].get(fn)
            img = cv.imread(img_file) if img_file else black_frame
            if img is None:
                img = black_frame

            overlay = img.copy()
            rect_x, rect_y = 5, 5
            rect_w, rect_h = 250, 50
            alpha = 0.6

            cv.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 0), -1)
            img = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            cam_index = self.camera_order.index(cam)
            delay = int(self.camera_delay_values[self.camera_order[cam_index]])
            cv.putText(img, f"Frame: {fn + delay}", (11, 41), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv.LINE_AA)
            cv.putText(img, f"Frame: {fn + delay}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

            return img
        except Exception as e:
            logging.error(f"video_generator.py: error loading frame {fn} for camera {cam}: {e}")
            self.errors = True
            return black_frame

