import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import glob
import os
import pathlib

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image



model_path = ".\\Prototyping\\pose_landmarker_heavy.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode = VisionRunningMode.IMAGE,
    num_poses = 2
)

with PoseLandmarker.create_from_options(options) as landmarker:
    curImg = 0
    if not (os.path.exists("annotatedImages")):
        os.makedirs("annotatedImages")
    if not (os.path.exists("landmarkersData")):
        os.makedirs("landmarkersData")
    localData = open("landmarkersData\\localLandmarkersData.txt", "w")
    globalData = open("landmarkersData\\globalLandmarkersData.txt", "w")
    for i in glob.glob(".\\frames\\*.jpg"):
        mp_img = mp.Image.create_from_file(i)
        pose_landmarker_result = landmarker.detect(mp_img)
        annotated_image = draw_landmarks_on_image(mp_img.numpy_view(), pose_landmarker_result)
        localData.write(str(pose_landmarker_result.pose_landmarks)+"\n")
        globalData.write(str(pose_landmarker_result.pose_world_landmarks)+"\n")
        print("creating image "+str(curImg))
        status = cv.imwrite(".\\annotatedImages\\AnnotatedFrame{f:{fill}{width}}.jpg".format(f=curImg, fill="0", width=4), cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))
        curImg += 1
