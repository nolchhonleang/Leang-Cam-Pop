Leang-Cam-Pop
Description
Leang-Cam-Pop is an augmented reality (AR) bubble-popping game developed by Nol Chhonleang. The game uses OpenCV for video processing and MediaPipe for hand tracking, allowing up to two players to pop colorful bubbles using their fingertips. The objective is to pop all bubbles before time runs out, with golden bubbles offering bonus points. The game features multiple levels, a scoring system, and interactive UI elements.
Features

Hand-tracking gameplay using MediaPipe
Supports up to two players
Dynamic bubble spawning with increasing difficulty
Interactive UI with buttons for Play, Next Level, Resume, Restart, and Quit
Victory animations and game-over screen
Keyboard controls: SPACE (start), S (pause/resume), N (next level), R (restart), Q (quit)

Requirements

Python 3.7+
Dependencies:
opencv-python
mediapipe
numpy



Installation

Clone the repository:git clone https://github.com/nolchhonleang/Leang-Cam-Pop.git
cd Leang-Cam-Pop


Install the required dependencies:pip install opencv-python mediapipe numpy


Ensure you have a webcam connected to your computer.

Usage
Run the game with:
python leang_cam_pop.py

Controls

Hand Gestures: Use your fingertips to pop bubbles or interact with buttons.
Keyboard:
SPACE: Start the game from the intro screen.
S: Pause or resume the game.
N: Advance to the next level (when paused).
R: Restart the game.
Q: Quit the game.



How to Play

Launch the game and point your webcam at your hands.
On the intro screen, use your fingertip to click the "Play" button or press SPACE.
Pop bubbles by touching them with your fingertip. Golden bubbles give extra points.
Clear all bubbles before the timer runs out to advance to the next level.
If the timer expires with bubbles remaining, the game ends, and the player with the highest score wins.
Use the pause menu to resume, go to the next level, restart, or quit.

Notes

Ensure good lighting for accurate hand tracking.
The game is optimized for a screen resolution of up to 1280x720.
If you encounter errors, verify that all dependencies are installed and your webcam is functional.

Author
Nol Chhonleang
License
This project is licensed under the MIT License.