# Video Generator Application
Useful tool to load pictures from a folder and create a video based on multiple cameras. The app fills missing frames with black ones for synchronization, the user can set starting and ending frames, camera layout, delays.

## Installation:
- You will need Python 3.13.
- Clone this repository, run the following to clone everything: `git clone https://github.com/Szaki73/Video-Generator-App.git`
- Preferably use a python virtual environment, and install dependencies via pip `pip install -r requirements.txt`
- If you prefer you can run this code from the terminal in a venv with `py main.py`

- If you prefer to create an executable from within a virtual environment, run: pyinstaller --onefile --windowed --name="Video Generator App" main.py
- This will generate several folders. Inside the folder named dist, you'll find Video Generator App.exe — this is all you need to run the app.

- Make sure to create a folder named output and a file called error.txt in the same directory as the executable.

## Usage

After starting the app we can break down the window into 3 parts: LEFT PANEL, TOP PANEL, GRID.

### LEFT PANEL

This where you can set up the inputs and output.

- Input Paths: you can give the program 5 different paths, at least 1 must be given, and all path must be valid.
- Output Path: The app has a deafault path it will be created if it is not existing. It is right next to the executabale.
- Output Name: It will be the name of the video. It must be give.
- Continue button: after the paths and the names are give pressing this will load the images. If anything is invalid we will get
an error message at the bottom of the left side.
- Stop Generator button: If we started a video generation we can stop it whit this. We will see a message at the bottom of the left panel: **Generation stopped**.
- Row count, Columns in first row, Columns in second row, Columns in third row: This are used to chnage the grid layout.

Note: We do not need to shape the grid to the cameras to get a nice video. For example if we have only two cameras and the layout is 3 by 3 the video wont be 3 by 3 it will be 1 by 2 so empty spaces will be ignored.

### TOP PANEL

This panel is for changing video setting.

- Framerate: Changes the framerate of the video.
- Start Frame: it is for changing the video length.
- End Frame: it is for changing the video length.
- Global Delay: this for running the video in the app so we can see if any camera is out of sync.
- Generate button: Pressing t will start the video generation. We can see the progress on the bottom of the left panel. If we had any error with the frames we will see this: **Video generated. Errors in the errors.txt.**, otherwise: **Video generated. No errors.**.

Note: Global delay changes all delay so after testing with it reset it to 0. Start frame must be less then End frame. We get a message at the bottom of the ledt panel is not.

### GRID

This is where we can see the loaded cameras. At start we see 16 empty cells. Each loaded cell will have a Delay spinbox under it. Clicking 2 loaded cells will change the cameras and the delay, this is how we can set the camera position for the video. Example: Let's say we have 2 cameras: cam1 and cam2. We set the delay on cam1 to 5. in the video we will see a frame wehere one of the cameras is on frame: 5 and the other will be on 0.