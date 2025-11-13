# Video Generator Application
Useful tool to load pictures from a folder and create a video based on multiple cameras. The app fills missing frames with black ones for synchronization, the user can set starting and ending frames, camera layout, delays.

## Installation:
- You will need Python 3.13 or later.
- Clone this repository, run the following to clone everything: `git clone https://github.com/Szaki73/Video-Generator-App.git`
- Preferably use a python virtual environment, and install dependencies via pip `pip install -r requirements.txt`
- If you prefer you can run this code from the terminal in a venv with `py main.py`

- If you prefer to create an executable from within a virtual environment, run: pyinstaller --onefile --windowed --name="Video Generator App" main.py
- This will generate several folders. Inside the folder named dist, you'll find Video Generator App.exe — this is all you need to run the app.

- Make sure to create a folder named output and a file called error.txt in the same directory as the executable.
