import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
import mediapipe as mp
import cv2

hand_landmarks = ['INDEX_FINGER_DIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_TIP',
                  'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_TIP',
                  'PINKY_DIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_TIP', 'RING_FINGER_DIP', 'RING_FINGER_MCP',
                  'RING_FINGER_PIP', 'RING_FINGER_TIP', 'THUMB_CMC', 'THUMB_IP', 'THUMB_MCP', 'THUMB_TIP', 'WRIST']
pose_landmarks = ['LEFT_ANKLE', 'LEFT_EAR', 'LEFT_ELBOW', 'LEFT_EYE', 'LEFT_EYE_INNER', 'LEFT_EYE_OUTER',
                  'LEFT_FOOT_INDEX', 'LEFT_HEEL', 'LEFT_HIP', 'LEFT_INDEX', 'LEFT_KNEE', 'LEFT_PINKY',
                  'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_WRIST', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'NOSE',
                  'RIGHT_ANKLE', 'RIGHT_EAR', 'RIGHT_ELBOW', 'RIGHT_EYE', 'RIGHT_EYE_INNER', 'RIGHT_EYE_OUTER',
                  'RIGHT_FOOT_INDEX', 'RIGHT_HEEL', 'RIGHT_HIP', 'RIGHT_INDEX', 'RIGHT_KNEE', 'RIGHT_PINKY',
                  'RIGHT_SHOULDER', 'RIGHT_THUMB', 'RIGHT_WRIST']


class KeypointExtractor:
    def __init__(self, num_labels, input_csv="videos_list_balanced.csv"):
        self.num_labels = num_labels
        self.input_csv = input_csv

    def extract_keypoint(self, video_path, label, actor):
        cap = cv2.VideoCapture(video_path)
        keypoint_dict = defaultdict(list)
        count = 0
        with mp.solutions.holistic.Holistic(min_detection_confidence=0.5,
                                            min_tracking_confidence=0.5) as holistic:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                count += 1
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                if results.right_hand_landmarks:
                    for idx, lm in enumerate(results.right_hand_landmarks.landmark):
                        keypoint_dict[f"{hand_landmarks[idx]}_right_x"].append(lm.x)
                        keypoint_dict[f"{hand_landmarks[idx]}_right_y"].append(lm.y)
                        keypoint_dict[f"{hand_landmarks[idx]}_right_z"].append(lm.z)
                else:
                    for idx in range(len(hand_landmarks)):
                        keypoint_dict[f"{hand_landmarks[idx]}_right_x"].append(0)
                        keypoint_dict[f"{hand_landmarks[idx]}_right_y"].append(0)
                        keypoint_dict[f"{hand_landmarks[idx]}_right_z"].append(0)
                if results.left_hand_landmarks:
                    for idx, lm in enumerate(results.left_hand_landmarks.landmark):
                        keypoint_dict[f"{hand_landmarks[idx]}_left_x"].append(lm.x)
                        keypoint_dict[f"{hand_landmarks[idx]}_left_y"].append(lm.y)
                        keypoint_dict[f"{hand_landmarks[idx]}_left_z"].append(lm.z)
                else:
                    for idx in range(len(hand_landmarks)):
                        keypoint_dict[f"{hand_landmarks[idx]}_left_x"].append(0)
                        keypoint_dict[f"{hand_landmarks[idx]}_left_y"].append(0)
                        keypoint_dict[f"{hand_landmarks[idx]}_left_z"].append(0)
                if results.pose_landmarks:
                    for idx, lm in enumerate(results.pose_landmarks.landmark):
                        keypoint_dict[f"{pose_landmarks[idx]}_x"].append(lm.x)
                        keypoint_dict[f"{pose_landmarks[idx]}_y"].append(lm.y)
                        keypoint_dict[f"{pose_landmarks[idx]}_z"].append(lm.z)
                else:
                    for idx in range(len(pose_landmarks)):
                        keypoint_dict[f"{pose_landmarks[idx]}_x"].append(0)
                        keypoint_dict[f"{pose_landmarks[idx]}_y"].append(0)
                        keypoint_dict[f"{pose_landmarks[idx]}_z"].append(0)
        keypoint_dict["frame"] = count
        keypoint_dict["video_path"] = video_path
        keypoint_dict["label"] = label
        keypoint_dict["actor"] = actor
        return keypoint_dict

    def process_videos(self):
        data = pd.read_csv(self.input_csv)
        keypoints_list = Parallel(n_jobs=-1)(
            delayed(self.extract_keypoint)(row['file'], row['label'], row['actor'])
            for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing videos", leave=False)
        )
        out_csv = f"vsl{self.num_labels}_keypoints.csv"
        pd.DataFrame(keypoints_list).to_csv(out_csv, index=False)
        return out_csv


class KeypointInterpolator:
    def __init__(self, num_labels, frames=80):
        self.num_labels = num_labels
        self.frames = frames
        self.HAND_IDENTIFIERS = [id + "_right" for id in hand_landmarks] + [id + "_left" for id in hand_landmarks]
        self.POSE_IDENTIFIERS = ["RIGHT_SHOULDER", "LEFT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW"]
        self.body_identifiers = self.HAND_IDENTIFIERS + self.POSE_IDENTIFIERS

    @staticmethod
    def find_index(array):
        for i, num in enumerate(array):
            if num != 0:
                return i

    def curl_skeleton(self, array):
        if sum(array) == 0:
            return array
        for i, location in enumerate(array):
            if location != 0:
                continue
            if i == 0 or i == len(array) - 1:
                continue
            if array[i + 1] != 0:
                array[i] = float((array[i - 1] + array[i + 1]) / 2)
            else:
                if sum(array[i:]) == 0:
                    continue
                j = self.find_index(array[i + 1:])
                array[i] = float(((1 + j) * array[i - 1] + 1 * array[i + 1 + j]) / (2 + j))
        return array

    def interpolate_keypoints(self, input_file, output_file):
        train_data = pd.read_csv(input_file)
        output_df = train_data.copy()
        for index, video in tqdm(train_data.iterrows(), total=train_data.shape[0], desc="Interpolating"):
            for identifier in self.body_identifiers:
                x_values = self.curl_skeleton(list(ast.literal_eval(video[identifier + "_x"])))
                y_values = self.curl_skeleton(list(ast.literal_eval(video[identifier + "_y"])))
                output_df.at[index, identifier + "_x"] = str(x_values)
                output_df.at[index, identifier + "_y"] = str(y_values)
        output_df.to_csv(output_file, index=False)
        return output_file

    def save_numpy(self, interpolated_csv):
        train_data = pd.read_csv(interpolated_csv)
        frames = self.frames
        data, labels = [], []
        for _, video in tqdm(train_data.iterrows(), total=train_data.shape[0], desc="Packing"):
            T = len(ast.literal_eval(video["INDEX_FINGER_DIP_right_x"]))
            current_row = np.empty(shape=(2, T, len(self.body_identifiers), 1))
            for index, identifier in enumerate(self.body_identifiers):
                x_vals = ast.literal_eval(video[identifier + "_x"])
                y_vals = ast.literal_eval(video[identifier + "_y"])
                current_row[0, :, index, :] = np.asarray(x_vals).reshape(T, 1)
                current_row[1, :, index, :] = np.asarray(y_vals).reshape(T, 1)
            if T < frames:
                target = np.zeros(shape=(2, frames, len(self.body_identifiers), 1))
                target[:, :T, :, :] = current_row
            else:
                target = current_row[:, :frames, :, :]
            data.append(target)
            labels.append(int(video["label"]))
        keypoint_data = np.stack(data, axis=0)
        label_data = np.stack(labels, axis=0)
        np.save(f'vsl{self.num_labels}_data_preprocess.npy', keypoint_data)
        np.save(f'vsl{self.num_labels}_label_preprocess.npy', label_data)

    def run(self):
        extractor = KeypointExtractor(self.num_labels)
        keypoints_csv = extractor.process_videos()
        out_csv = f"vsl{self.num_labels}_interpolated_keypoints.csv"
        interpolated_csv = self.interpolate_keypoints(keypoints_csv, out_csv)
        self.save_numpy(interpolated_csv)


if __name__ == "__main__":
    #num_labels = 10
    interpolator = KeypointInterpolator(num_labels=num_labels, frames=80)
    interpolator.run()
