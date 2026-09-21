import cv2
import mediapipe as mp
import random
import streamlit as st
import av
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(page_title="Nose-Driven Bird Game", page_icon="🐦")
st.title("🐦 Nose-Driven Bird Game")
st.write("Use your nose position to fly through the green pipes!")

base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)
detector = vision.PoseLandmarker.create_from_options(options)

bird_path = "assets/bird.png"
pipe_cap_path = "assets/pipe_cap.png"
pipe_body_path = "assets/pipe_body.png"
curr_path = os.path.dirname(__file__)

bird_full_path = os.path.join(curr_path, bird_path)
pipe_cap_full_path = os.path.join(curr_path, pipe_cap_path)
pipe_body_full_path = os.path.join(curr_path, pipe_body_path)

bird = cv2.imread(bird_full_path, cv2.IMREAD_UNCHANGED)
pipe_cap = cv2.imread(pipe_cap_full_path, cv2.IMREAD_UNCHANGED)
pipe_body = cv2.imread(pipe_body_full_path, cv2.IMREAD_UNCHANGED)

def overlay_sprite(background, sprite, x, y):
    h, w = sprite.shape[:2]

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, background.shape[1])
    y2 = min(y + h, background.shape[0])

    if x1 >= x2 or y1 >= y2:
        return

    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    sprite_crop = sprite[sy1:sy2, sx1:sx2]

    if sprite.shape[2] == 4:
        sprite_rgb = sprite_crop[:, :, :3]
        alpha = sprite_crop[:, :, 3] / 255.0

        for c in range(3):
            background[y1:y2, x1:x2, c] = (
                alpha * sprite_rgb[:, :, c] +
                (1 - alpha) * background[y1:y2, x1:x2, c]
            )
    else:
        background[y1:y2, x1:x2] = sprite_crop

def draw_body(background, body, x, y, height):
    body_h = body.shape[0]

    current_y = y

    while current_y < y + height:
        remaining = y + height - current_y
        draw_h = min(body_h, remaining)
        piece = body[:draw_h]
        overlay_sprite(background, piece, x, current_y)
        current_y += draw_h

def rotate_sprite(sprite, angle):
    h, w = sprite.shape[:2]

    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(sprite, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

class GameState:
    def __init__(self):
        self.bird_x = 100
        self.bird_y = 200
        self.previous_bird_y = self.bird_y
        self.bird_radius = 20

        self.pipe_width = 70
        self.pipe_gap = 100
        self.pipe_speed = 8

        self.pipes = [
            Pipe(600, random.randint(50, 250)),
            Pipe(900, random.randint(50, 250)),
            Pipe(1200, random.randint(50, 250))
        ]

        self.score = 0
        self.game_over = False

class Pipe:
    def __init__(self, x, top_height):
        self.x = x
        self.top_height = top_height

if "high_score" not in st.session_state:
    st.session_state.high_score = 0

if "game" not in st.session_state:
    st.session_state.game = GameState()

game = st.session_state.game

bird = cv2.resize(bird, (2 * game.bird_radius, 2 * game.bird_radius))

cap_h, cap_w = pipe_cap.shape[:2]
pipe_cap_h = round(cap_h * game.pipe_width / cap_w)
pipe_cap_resized = cv2.resize(pipe_cap, (game.pipe_width, pipe_cap_h))
top_pipe_cap = cv2.flip(pipe_cap_resized, 0)

body_h, body_w = pipe_body.shape[:2]
pipe_body_h = round(body_h * game.pipe_width / body_w)
pipe_body_resized = cv2.resize(pipe_body, (game.pipe_width, pipe_body_h))

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    try:
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        if detection_result.pose_landmarks and not game.game_over:
            first_pose = detection_result.pose_landmarks[0]
            nose_y = first_pose[0].y
            game.previous_bird_y = game.bird_y
            game.bird_y = int(nose_y * h)
    except Exception:
        pass

    if not game.game_over:
        for pipe in game.pipes:
            pipe.x -= game.pipe_speed

            if pipe.x < -game.pipe_width:
                rightmost_x = max(p.x for p in game.pipes)

                pipe.x = rightmost_x + 300
                pipe.top_height = random.randint(50, h - game.pipe_gap - 50)
                game.score += 1

        for pipe in game.pipes:
            if pipe.x < (game.bird_x + game.bird_radius) < (pipe.x + game.pipe_width):
                if game.bird_y - game.bird_radius < pipe.top_height or game.bird_y + game.bird_radius > pipe.top_height + game.pipe_gap:
                    game.game_over = True

    for pipe in game.pipes:
        overlay_sprite(img, pipe_cap_resized, pipe.x, pipe.top_height + game.pipe_gap)
        overlay_sprite(img, top_pipe_cap, pipe.x, pipe.top_height - pipe_cap_h)

        draw_body(img, pipe_body_resized, pipe.x, 0, pipe.top_height - pipe_cap_h)
        draw_body(img, pipe_body_resized, pipe.x, pipe.top_height + game.pipe_gap + pipe_cap_h, h - (pipe.top_height + game.pipe_gap + pipe_cap_h))

    dy = game.bird_y - game.previous_bird_y
    angle = max(-30, min(30, -dy * 2))
    rotated_bird = rotate_sprite(bird, angle)
    overlay_sprite(img, rotated_bird, game.bird_x - rotated_bird.shape[1] // 2, game.bird_y - rotated_bird.shape[0] // 2)

    if game.game_over:
        cv2.putText(img, "GAME OVER", (w // 2 - 130, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
    else:
        cv2.putText(img, f"Score: {game.score}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="pose-game-full",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False}
)

@st.fragment(run_every="500ms")
def game_ui():
    if game.game_over:
        st.session_state.high_score = max(st.session_state.high_score, game.score)

        st.write(f"Score: {game.score}")
        st.write(f"High Score: {st.session_state.high_score}")

        if st.button("🔄 Restart 🔄"):
            st.session_state.game = GameState()
            st.rerun()

game_ui()