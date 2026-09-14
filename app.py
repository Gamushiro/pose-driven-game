import cv2
import mediapipe as mp
import numpy as np
import random
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode

st.set_page_config(page_title="Pose-Driven Flappy Bird", page_icon="🐦")
st.title("🐦 ML Pose-Driven Flappy Bird")
st.write("Use your head position to fly through the green pipes!")

class PoseGameTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_pose = mp.solution.pose
        self.pose = self.mp_pose.Pose(min_detecion_confidence=0.7, min_tracking_confidence=0.7)

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

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks and not self.game_over:
            nose_y = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            self.bird_y = int(nose_y * h)

        if not self.game_over:
            self.pipe_x -= self.pipe_speed

            if self.pipe_x < -self.width:
                self.pipe_x = w
                self.pipe_top_height = random.randint(50, h - self.pipe_gap - 50)
                self.score += 1


