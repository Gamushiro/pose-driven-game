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
st.title("🐦 ML Nose-Driven Bird Game")
st.write("Use your nose position to fly through the green pipes!")

base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)
detector = vision.PoseLandmarker.create_from_options(options)

bird_path = "assets/bird.png"
pipe_path = "assets/pipe.png"
curr_path = os.path.dirname(__file__)
bird_full_path = os.path.join(curr_path, bird_path)
pipe_full_path = os.path.join(curr_path, pipe_path)
bird = cv2.imread(bird_full_path, cv2.IMREAD_UNCHANGED)
pipe = cv2.imread(pipe_full_path, cv2.IMREAD_UNCHANGED)

bird = cv2.resize(bird, (40, 40))

def overlay_sprite(background, sprite, x, y):
    h, w = sprite.shape[:2]

    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return

    if sprite.shape[2] == 4:
        sprite_rgb = sprite[:, :, :3]
        alpha = sprite[:, :, 3] / 255.0

        for c in range(3):
            background[y:y+h, x:x+w, c] = (
                alpha * sprite_rgb[:, :, c] +
                (1 - alpha) * background[y:y+h, x:x+w, c]
            )
    else:
        background[y:y+h, x:x+w] = sprite

class GameState:
    def __init__(self):
        self.bird_x = 100
        self.bird_y = 200
        self.bird_radius = 20

        self.pipe_x = 600
        self.pipe_width = 70
        self.pipe_gap = 100
        self.pipe_speed = 8
        self.pipe_top_height = random.randint(50, 250)

        self.score = 0
        self.game_over = False

if "high_score" not in st.session_state:
    st.session_state.high_score = 0

if "game" not in st.session_state:
    st.session_state.game = GameState()

game = st.session_state.game

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
            game.bird_y = int(nose_y * h)
    except Exception:
        pass

    if not game.game_over:
        game.pipe_x -= game.pipe_speed

        if game.pipe_x < -game.pipe_width:
            game.pipe_x = w
            game.pipe_top_height = random.randint(50, h - game.pipe_gap - 50)
            game.score += 1

        if game.pipe_x < (game.bird_x + game.bird_radius) < (game.pipe_x + game.pipe_width):
            if game.bird_y - game.bird_radius < game.pipe_top_height or game.bird_y + game.bird_radius > game.pipe_top_height + game.pipe_gap:
                game.game_over = True

    top_pipe = cv2.resize(pipe, (game.pipe_width, game.pipe_top_height))
    top_pipe = cv2.flip(top_pipe, 0)
    overlay_sprite(img, top_pipe, game.pipe_x, 0)

    bottom_pipe_y = game.pipe_top_height + game.pipe_gap
    bottom_pipe_height = h - bottom_pipe_y
    bottom_pipe = cv2.resize(pipe, (game.pipe_width, bottom_pipe_height))
    overlay_sprite(img, bottom_pipe, game.pipe_x, bottom_pipe_y)

    bird_x = game.bird_x - game.bird_radius
    bird_y = game.bird_y - game.bird_radius
    overlay_sprite(img, bird, bird_x, bird_y)

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
        st.write(f"🏆High Score: {st.session_state.high_score}")

        if st.button("🔄 Restart 🔄"):
            st.session_state.game = GameState()
            st.rerun()

game_ui()