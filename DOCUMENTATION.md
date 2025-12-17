# DOCUMENTATION

## main.py

This is the enty point for the app. It makes running easier. And sets the window size.

## clickable_lable.py

This is a PySide6 QLabel class for handling clicking.

## default_spinbox.py

This is a PySide6 QSpinBox class for showing text in an integer spinbox. Start Frame and End Frame on default are 0 which can be missleading with this class i set the 0 values to show The frist and The last.

## generator_window.py

`if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))`

This is for handling executable and terminal running. If the users uses executable then abspath won't work.

`log_file = os.path.join(base_path, "error.txt")

logging.basicConfig(
    filename=log_file,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)`

This is for logging errors in the error.txt.

### Fields

- self.input_entries = []: Holds the inputs text fields.
- self.input_buttons = []: Holds the browse buttons for the inputs.
- self.inputs = []: Holds all of the input path.
- self.output_path = os.path.join(base_path, "output"): Holds the output path, on default it points to the output folder.
- self.output_name = "output": The output video name: on default it is `output`
- self.row = 4: number of rows in the gird, 3 on default.
- self.columns = [4, 4, 4, 4]: number of column for each row in the gird, each is 3 on default.
- self.framerate = 15: the video framerate, 15 on default.
- self.start_frame = 0: start frame of the video, it is 0 on default which means the first frame.
- self.end_frame = 0: end frame of the video, it is 0 on default which means the last frame.
- self.camera_grid = []: the cameras for showing them in the grid.
- self.camera_positions = {}: the positions of the cameras in the grid.
- self.camera_order = []: the order of the cameras in the grid from left to right, top to bottom.
- self.camera_frames = {}: dictionary of cmaeras and the list of frames for each camera in order.
- self.frame_numbers = set(): holding all unique framenumbers.
- self.camera_delay_values = {}: holding the delay values for each camera in a dictionary.
- self.delay_vars = {}: holding the delay vals.
- self.stop_generator = False: bool for checking if generation was stopped or not.
- self.errors = False: bool for nowing if we had an error or not.

- self.frame_data = {}: holding the frames for each camera
- self.image_width = None: holding image width
- self.image_height = None: holding image height
- self.video_width = None: holding video width
- self.video_height = None: holding video height
- self.image_labels = {}: saving the camera labels for swapping
- self.selected_cam = None: holding a camera for swapping
- self.black_frame = None: black frame for filling missing frames
- self.scale_factor = 1.0: for scaling down images if the video would be wider than 4096 pixel

### Methods

- showEvent(self, event): this is for showing the grid properly at the start of the app.
- def init_ui(self): initializes the whole window.
- def browse_input(self): browsing the input path.
- def browse_lidar_input(self): browsing the lidar input path.
- def browse_output(self): browsing the output path.
- def create_grid_view(self): creates the grid for holding the cameras and showing the delay spinbox under them.
- def update_row(self, value): changes the row value and updates the grid.
- def def update_column(self, index, value):: changes the column value for a row and updates the grid.
- def loading_grid_view(self): validates the paths and the output name. Loads the images in the grid.
- def stop_gen(self): sets self.stop_generator to be True.
- def update_grid(self): updates the whole grid when a column or a row changes.
- def load_frames(self): loads frames for the app.
- def get_image_sizes(self): gets the images sizes.
- def get_frame_image(self, cam, frame_index): gets a frmae for a camera, return black image if frame is missing.
- get_black_image(self): creats black image for app.
- resizeEvent(self, event): overrides the default resize event to resize the grid for better visibility.
- handle_click(self, cam): handles clicks. Swaps delay values and cameras in the app.
- update_frame(self, cam): updates one camera in the app.
- def on_delay_change(self, cam, delay_val): Changes a camera in the app based on the delay.
- def on_global_delay_change(self, new_global_delay): Changes all camers for testing delays.
- def generate_video(self): generates video. Checks if generator was stopped. wirtes frames to video, writes errors in file.
- def sort_images(self): loads the frmes for the generation part based on camera_order.
-  def get_video_height_and_video_width(self): sets the video shapes based on camera numbers and the row and columns values.
- def load_and_set_frame(self, cam, fn, camera_frames, height, width, black_frame, camera_order, camera_delay_values):
Get a frames wirtes frames number on it, or return a black frame is frame is missig.
- 