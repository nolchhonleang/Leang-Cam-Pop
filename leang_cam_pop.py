# Leang-Cam-Pop
# Author: Nol Chhonleang
# Description: An augmented reality bubble-popping game using OpenCV and MediaPipe for hand tracking.
# Created: August 27, 2025

import numpy as np
import random
import time
from dataclasses import dataclass
import tkinter as tk
import cv2

try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("\n[!] Missing dependency: mediapipe\n    Install with: pip install mediapipe opencv-python numpy\n")

# ------------------------ Visual Helpers ------------------------
NEON_COLORS = [
    (0, 255, 255), (0, 255, 128), (255, 0, 255), (255, 128, 0),
    (0, 128, 255), (255, 0, 128), (128, 255, 0), (128, 0, 255)
]
GOLD_COLOR = (255, 215, 0)  # Golden bubble color
UI_WHITE = (245, 245, 245)
UI_BLACK = (25, 25, 25)
UI_SHADOW = (0, 0, 0)
HOVER_COLOR = (200, 200, 200)  # Brighter hover color
BUTTON_COLORS = {
    'play': (0, 255, 255),  # Cyan
    'next': (0, 255, 0),    # Green
    'resume': (0, 128, 255),# Blue
    'restart': (255, 165, 0), # Orange
    'quit': (255, 0, 0)     # Red
}

def draw_soft_text(img, text, org, scale=0.9, color=UI_WHITE, thickness=2):
    x, y = org
    cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, scale, UI_SHADOW, thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_chip(img, text, anchor=(15, 35), pad=10, bg=(40, 40, 40)):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    x, y = anchor
    cv2.rectangle(img, (x- pad, y - h - pad//2), (x + w + pad, y + pad//2), bg, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x- pad, y - h - pad//2), (x + w + pad, y + pad//2), (60,60,60), 2, cv2.LINE_AA)
    draw_soft_text(img, text, (x, y), 0.65, UI_WHITE, 2)

def draw_button(img, text, top_left, size, action, is_hovered=False):
    x, y = top_left
    w, h = size
    bg = BUTTON_COLORS.get(action, (100, 100, 100))
    if is_hovered:
        bg = tuple(min(255, c + 60) for c in bg)  # Slightly brighter for hover
        w, h = int(w * 1.1), int(h * 1.1)  # Enlarge on hover
        x, y = x - int(w * 0.05), y - int(h * 0.05)  # Center adjustment
    # Draw glow effect for buttons
    overlay = img.copy()
    cv2.rectangle(overlay, (x-5, y-5), (x + w + 5, y + h + 5), bg, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    # Draw shadow
    cv2.rectangle(img, (x+3, y+3), (x + w + 3, y + h + 3), UI_SHADOW, -1, cv2.LINE_AA)
    # Draw button
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), 2, cv2.LINE_AA)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    text_w, text_h = text_size
    text_x = x + (w - text_w) // 2
    text_y = y + (h + text_h) // 2
    # Consistent text scale and thickness for all buttons
    draw_soft_text(img, text, (text_x, text_y), 0.8, UI_WHITE, 2)
    return (x, y, x + w, y + h)

def draw_star_icon(img, center, r, color=UI_WHITE):
    cx, cy = center
    points = []
    for i in range(10):
        angle = i * np.pi / 5 + np.pi / 2
        radius = r if i % 2 == 0 else r * 0.5
        x = int(cx + radius * np.cos(angle))
        y = int(cy - radius * np.sin(angle))
        points.append((x, y))
    cv2.polylines(img, [np.array(points, np.int32)], True, color, 2, cv2.LINE_AA)
    cv2.fillPoly(img, [np.array(points, np.int32)], (255, 255, 255, 50), cv2.LINE_AA)

def draw_victory_animation(frame, width, height, t):
    alpha = 0.5 * (1 + np.sin(t * 5))  # Pulsating effect
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (int(width * 0.8), height), (0, 255, 0, int(alpha * 60)), -1)
    cv2.addWeighted(overlay, alpha * 0.4, frame, 1 - alpha * 0.4, 0, frame)
    # Smaller, more fitting "Level Complete!" text
    draw_soft_text(frame, "Level Complete!", (int(width*0.35), int(height*0.5)), 0.9, (0, 255, 0), 2)

def draw_right_panel(frame, width, height):
    overlay = frame.copy()
    cv2.rectangle(overlay, (int(width * 0.8), 0), (width, height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

# ------------------------ Game Objects ------------------------
@dataclass
class Bubble:
    x: float
    y: float
    r: int
    vx: float
    vy: float
    color: tuple
    is_golden: bool = False
    popped: bool = False
    pop_t: float = 0.0

    def update(self, w, h):
        if self.popped:
            return
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.05
        if self.x - self.r < 0:
            self.x = self.r
            self.vx *= -0.9
        if self.x + self.r > w:
            self.x = w - self.r
            self.vx *= -0.9
        if self.y - self.r < 0:
            self.y = self.r
            self.vy *= -0.9
        if self.y + self.r > h:
            self.y = h - self.r
            self.vy *= -0.9

    def draw(self, frame):
        if not self.popped:
            cv2.circle(frame, (int(self.x), int(self.y)), self.r, self.color, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(self.x), int(self.y)), self.r, UI_WHITE, 2, cv2.LINE_AA)
            draw_star_icon(frame, (int(self.x), int(self.y)), self.r // 2, UI_WHITE)
        else:
            age = time.time() - self.pop_t
            rr = int(self.r + age * 200)
            alpha = max(0, 1.0 - age * 2.0)
            if alpha > 0:
                overlay = frame.copy()
                cv2.circle(overlay, (int(self.x), int(self.y)), rr, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            print(f"[SFX] Pop sound for {'golden' if self.is_golden else 'regular'} bubble")

    def try_pop(self, px, py):
        if self.popped:
            return 0
        dist = ((self.x - px)**2 + (self.y - py)**2)**0.5
        if dist <= self.r * 0.9:
            self.popped = True
            self.pop_t = time.time()
            return 50 if self.is_golden else 10
        return 0

# ------------------------ Game Engine ------------------------
class LeangPopGame:
    def __init__(self, camera_index=0):
        root = tk.Tk()
        width = min(root.winfo_screenwidth(), 1280)
        height = min(root.winfo_screenheight(), 720)
        root.destroy()

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.bubbles = []
        self.player_scores = [0, 0]
        self.player_cursors = [None, None]
        self.level = 1
        self.state = 'INTRO'
        self.start_time = time.time()
        self.level_time = 45
        self.max_players = 2
        self.button_rects = {}
        self.victory_time = 0

    def spawn_wave(self, level):
        self.bubbles = []
        n = min(10 + level * 5, 50)
        max_speed = 2.2 + level * 0.5
        min_radius = max(15, 18 - level * 2)
        max_radius = max(30, 42 - level * 3)
        self.level_time = max(20, 45 - level * 5)
        
        attempts = 0
        max_attempts = n * 10
        while len(self.bubbles) < n and attempts < max_attempts:
            r = random.randint(min_radius, max_radius)
            x = random.randint(r + 10, int(self.width * 0.8) - r - 10)
            y = random.randint(r + 10, self.height - r - 10)
            overlaps = False
            for b in self.bubbles:
                dist = ((x - b.x)**2 + (y - b.y)**2)**0.5
                if dist < (r + b.r) * 1.2:
                    overlaps = True
                    break
            if not overlaps:
                vx = random.uniform(-max_speed, max_speed)
                vy = random.uniform(-max_speed * 0.75, max_speed * 0.75)
                is_golden = random.random() < 0.1 and level > 1
                color = GOLD_COLOR if is_golden else random.choice(NEON_COLORS)
                if is_golden:
                    vx *= 1.5
                    vy *= 1.5
                self.bubbles.append(Bubble(x, y, r, vx, vy, color, is_golden))
            attempts += 1
        self.start_time = time.time()

    def fingertip_from_landmarks(self, frame, hand_landmarks):
        h, w, _ = frame.shape
        idx_tip = hand_landmarks.landmark[8]
        x = int(idx_tip.x * w)
        y = int(idx_tip.y * h)
        return x, y

    def smooth_cursor(self, x, y, player_idx, alpha=0.2):
        if self.player_cursors[player_idx] is None:
            self.player_cursors[player_idx] = (x, y)
        sx = int(self.player_cursors[player_idx][0] * (1 - alpha) + x * alpha)
        sy = int(self.player_cursors[player_idx][1] * (1 - alpha) + y * alpha)
        self.player_cursors[player_idx] = (sx, sy)
        return sx, sy

    def draw_intro(self, frame):
        draw_right_panel(frame, self.width, self.height)
        msg1 = "LEANG-CAM-POP!"
        msg2 = "Pop all bubbles before time runs out!"
        msg3 = "Controls: Click Play or SPACE=start, s=pause, click buttons or n=next, r=restart, q=quit"
        draw_soft_text(frame, msg1, (int(self.width*0.25), int(self.height*0.4)), 2.0, (0,255,255), 4)
        draw_soft_text(frame, msg2, (int(self.width*0.2), int(self.height*0.55)), 1.0, UI_WHITE, 3)
        draw_soft_text(frame, msg3, (int(self.width*0.1), int(self.height*0.7)), 0.6, (200,200,200), 1)
        self.button_rects = {}
        fingertips = self.get_fingertips(frame)
        is_hovered = any(cx is not None and self.check_button_click(cx, cy) == 'play' for cx, cy, _ in fingertips)
        button_x = int(self.width * 0.82)
        button_y = int(self.height * 0.3)
        self.button_rects['play'] = draw_button(frame, "Play", (button_x, button_y), (200, 60), 'play', is_hovered)

    def draw_hud(self, frame):
        remaining = max(0, int(self.level_time - (time.time() - self.start_time)))
        bubbles_left = sum(1 for b in self.bubbles if not b.popped)
        for i in range(self.max_players):
            draw_chip(frame, f"Player {i+1} Score: {self.player_scores[i]}", (15, 40 + i*40))
        draw_chip(frame, f"Level: {self.level}", (15, 40 + self.max_players*40))
        draw_chip(frame, f"Time: {remaining}s", (15, 80 + self.max_players*40))
        draw_chip(frame, f"Bubbles Left: {bubbles_left}", (15, 120 + self.max_players*40))

    def draw_pause_menu(self, frame):
        draw_right_panel(frame, self.width, self.height)
        self.button_rects = {}
        button_width, button_height = 200, 60
        button_spacing = 120
        fingertips = self.get_fingertips(frame)
        button_x = int(self.width * 0.82)
        self.button_rects['next'] = draw_button(frame, "Next Level", (button_x, int(self.height * 0.2)), 
                                               (button_width, button_height), 'next',
                                               is_hovered=any(self.check_button_click(cx, cy) == 'next' for cx, cy, _ in fingertips))
        self.button_rects['resume'] = draw_button(frame, "Resume", (button_x, int(self.height * 0.2 + button_spacing)), 
                                                 (button_width, button_height), 'resume',
                                                 is_hovered=any(self.check_button_click(cx, cy) == 'resume' for cx, cy, _ in fingertips))
        self.button_rects['restart'] = draw_button(frame, "Restart", (button_x, int(self.height * 0.2 + 2 * button_spacing)), 
                                                  (button_width, button_height), 'restart',
                                                  is_hovered=any(self.check_button_click(cx, cy) == 'restart' for cx, cy, _ in fingertips))
        self.button_rects['quit'] = draw_button(frame, "Quit", (button_x, int(self.height * 0.2 + 3 * button_spacing)), 
                                                (button_width, button_height), 'quit',
                                                is_hovered=any(self.check_button_click(cx, cy) == 'quit' for cx, cy, _ in fingertips))
        if time.time() - self.victory_time < 2:
            draw_victory_animation(frame, self.width, self.height, time.time() - self.victory_time)

    def draw_game_over(self, frame):
        draw_right_panel(frame, self.width, self.height)
        self.button_rects = {}
        button_width, button_height = 200, 60
        button_spacing = 120
        fingertips = self.get_fingertips(frame)
        button_x = int(self.width * 0.82)
        self.button_rects['restart'] = draw_button(frame, "Restart", (button_x, int(self.height * 0.3)), 
                                                  (button_width, button_height), 'restart',
                                                  is_hovered=any(self.check_button_click(cx, cy) == 'restart' for cx, cy, _ in fingertips))
        self.button_rects['quit'] = draw_button(frame, "Quit", (button_x, int(self.height * 0.3 + button_spacing)), 
                                                (button_width, button_height), 'quit',
                                                is_hovered=any(self.check_button_click(cx, cy) == 'quit' for cx, cy, _ in fingertips))
        # Calculate text size to center "GAME OVER"
        text = "GAME OVER"
        scale = 1.0
        thickness = 2
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        text_w, text_h = text_size
        text_x = (self.width - text_w) // 2
        text_y = (self.height + text_h) // 2
        draw_soft_text(frame, text, (text_x, text_y), scale, (255, 0, 0), thickness)
        # Determine and display the winner
        winner_text = "It's a Tie!" if self.player_scores[0] == self.player_scores[1] else \
                      f"Player 1 Wins!" if self.player_scores[0] > self.player_scores[1] else "Player 2 Wins!"
        winner_scale = 0.8
        winner_thickness = 2
        winner_size, winner_baseline = cv2.getTextSize(winner_text, cv2.FONT_HERSHEY_SIMPLEX, winner_scale, winner_thickness)
        winner_w, winner_h = winner_size
        winner_x = (self.width - winner_w) // 2
        winner_y = text_y + text_h + 20  # Position below "GAME OVER" with some spacing
        draw_soft_text(frame, winner_text, (winner_x, winner_y), winner_scale, UI_WHITE, winner_thickness)
        self.draw_hud(frame)

    def get_fingertips(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        fingertips = []
        if res.multi_hand_landmarks:
            for i, handLm in enumerate(res.multi_hand_landmarks):
                if i >= self.max_players:
                    break
                cx, cy = self.fingertip_from_landmarks(frame, handLm)
                cx, cy = self.smooth_cursor(cx, cy, i)
                fingertips.append((cx, cy, i))
        return fingertips

    def check_button_click(self, cx, cy):
        for action, (x1, y1, x2, y2) in self.button_rects.items():
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return action
        return None

    def update_and_draw_bubbles(self, frame):
        alive = 0
        for b in self.bubbles:
            b.update(int(self.width * 0.8), self.height)
            b.draw(frame)
            if not b.popped:
                alive += 1
        return alive

    def try_pop_with_cursor(self, cx, cy, player_idx):
        points = 0
        for b in self.bubbles:
            points += b.try_pop(cx, cy)
        if points > 0:
            remaining = max(0, self.level_time - (time.time() - self.start_time))
            multiplier = 1 + remaining / self.level_time
            self.player_scores[player_idx] += int(points * multiplier)
            print(f"[SFX] Score +{int(points * multiplier)} for Player {player_idx + 1}")

    def run(self):
        if not self.cap.isOpened():
            raise SystemExit("[!] Could not open camera.")

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            fingertips = self.get_fingertips(frame)
            for cx, cy, i in fingertips:
                color = NEON_COLORS[i % len(NEON_COLORS)]
                cv2.circle(frame, (cx, cy), 10, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 14, UI_BLACK, 2, cv2.LINE_AA)
                draw_soft_text(frame, str(i+1), (cx + 15, cy - 15), 0.7, color, 2)

            if self.state == 'INTRO':
                self.draw_intro(frame)
                for cx, cy, player_idx in fingertips:
                    action = self.check_button_click(cx, cy)
                    if action == 'play':
                        self.spawn_wave(self.level)
                        self.state = 'PLAY'
            elif self.state == 'PLAY':
                alive = self.update_and_draw_bubbles(frame)
                for cx, cy, player_idx in fingertips:
                    self.try_pop_with_cursor(cx, cy, player_idx)
                self.draw_hud(frame)
                if time.time() - self.start_time >= self.level_time:
                    self.state = 'GAME_OVER' if alive > 0 else 'PAUSE'
                elif alive == 0:
                    self.state = 'PAUSE'
                    self.victory_time = time.time()
            elif self.state == 'PAUSE':
                self.draw_hud(frame)
                self.draw_pause_menu(frame)
                for cx, cy, player_idx in fingertips:
                    action = self.check_button_click(cx, cy)
                    if action == 'next':
                        self.level += 1
                        self.spawn_wave(self.level)
                        self.state = 'PLAY'
                    elif action == 'resume':
                        self.state = 'PLAY'
                        self.start_time = time.time()
                    elif action == 'restart':
                        self.player_scores = [0, 0]
                        self.level = 1
                        self.spawn_wave(self.level)
                        self.state = 'PLAY'
                    elif action == 'quit':
                        self.cap.release()
                        cv2.destroyAllWindows()
                        return
            elif self.state == 'GAME_OVER':
                self.draw_game_over(frame)
                for cx, cy, player_idx in fingertips:
                    action = self.check_button_click(cx, cy)
                    if action == 'restart':
                        self.player_scores = [0, 0]
                        self.level = 1
                        self.spawn_wave(self.level)
                        self.state = 'PLAY'
                    elif action == 'quit':
                        self.cap.release()
                        cv2.destroyAllWindows()
                        return

            cv2.imshow('Leang-Pop! — AR Camera Game', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                if self.state == 'INTRO':
                    self.spawn_wave(self.level)
                    self.state = 'PLAY'
            elif key == ord('s'):
                if self.state == 'PAUSE':
                    self.state = 'PLAY'
                    self.start_time = time.time()
                elif self.state == 'PLAY':
                    self.state = 'PAUSE'
            elif key == ord('r'):
                self.player_scores = [0, 0]
                self.level = 1
                self.spawn_wave(self.level)
                self.state = 'PLAY'
            elif key == ord('n'):
                if self.state == 'PAUSE':
                    self.level += 1
                    self.spawn_wave(self.level)
                    self.state = 'PLAY'

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        game = LeangPopGame(camera_index=0)
        game.run()
    except Exception as e:
        print("\n[!] Error:", e)
        print("\nSetup steps:\n  pip install opencv-python mediapipe numpy\n  Run: python leang_cam_pop.py\n")