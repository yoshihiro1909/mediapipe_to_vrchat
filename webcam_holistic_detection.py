from re import M
import mediapipe as mp
import cv2
import time
import osc_client
from rich import print
import argparse
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import pickle
import mmap
from scipy.spatial import distance
from logging import getLogger, INFO, StreamHandler, FileHandler, lastResort
import time

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)


class PoseLogger:
    def __init__(self):
        fh = FileHandler(f"pose_" + ".csv")
        logger.addHandler(fh)

    def on_message(self, msg):
        logger.info("{}".format(msg))


pose_logger = PoseLogger()

pos = np.zeros((33, 3))

ip = "127.0.0.1"
port = "39570"
# port="5005"
osc_client_inst = osc_client.MonoBehaviour(ip, port)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)

    # parser.add_argument("--k_new_param", type=float, default=1.0)
    parser.add_argument("--k_new_param", type=float, default=0.9)

    # parser.add_argument("--k_filename", type=str, default="K.csv")
    # parser.add_argument("--d_filename", type=str, default="d.csv")
    parser.add_argument("--k_filename", type=str, default="K_fisheye.csv")
    parser.add_argument("--d_filename", type=str, default="d_fisheye.csv")

    args = parser.parse_args()

    return args


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=1, color=(0, 255, 0))
mark_drawing_spec = mp_drawing.DrawingSpec(
    thickness=2, circle_radius=2, color=(0, 0, 255)
)

# camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera = cv2.VideoCapture(0)
args = get_args()

cap_device = args.device
filepath = args.file
cap_width = args.width
cap_height = args.height

k_new_param = args.k_new_param

k_filename = args.k_filename
d_filename = args.d_filename

camera.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

camera_mat = np.loadtxt(k_filename, delimiter=",")
dist_coef = np.loadtxt(d_filename, delimiter=",")

new_camera_mat = camera_mat.copy()
new_camera_mat[(0, 1), (0, 1)] = k_new_param * new_camera_mat[(0, 1), (0, 1)]

with mp_holistic.Holistic(
    min_detection_confidence=0.5, static_image_mode=False
) as holistic_detection:
    while camera.isOpened():
        start_time = time.time()
        # time.sleep(1 / 30)

        success, image = camera.read()
        image = cv2.flip(image, 1)
        undistort_image = cv2.fisheye.undistortImage(
            image,
            camera_mat,
            D=dist_coef,
            Knew=new_camera_mat,
        )

        image = undistort_image.copy()
        if not success:
            print("empty camera frame")
            continue
        # image = cv2.resize(image, dsize=None, fx=0.3, fy=0.3)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = holistic_detection.process(rgb_image)
        # image = image -1000
        if (
            results.face_landmarks
            or results.pose_landmarks
            or results.left_hand_landmarks
            or results.right_hand_landmarks
        ):
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.face_landmarks,
                connections=mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mesh_drawing_spec,
            )
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.pose_landmarks,
                connections=mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mark_drawing_spec,
                connection_drawing_spec=mesh_drawing_spec,
            )
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.left_hand_landmarks,
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mark_drawing_spec,
                connection_drawing_spec=mesh_drawing_spec,
            )
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.right_hand_landmarks,
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mark_drawing_spec,
                connection_drawing_spec=mesh_drawing_spec,
            )

            # print(f'=============================')
            real_shoulder_width = 0.35
            x27 = results.pose_landmarks.landmark[11].x * cap_width
            y27 = results.pose_landmarks.landmark[11].y * cap_height
            z27 = results.pose_landmarks.landmark[11].z

            x28 = results.pose_landmarks.landmark[12].x * cap_width
            y28 = results.pose_landmarks.landmark[12].y * cap_height
            z28 = results.pose_landmarks.landmark[12].z

            hip_x = (x27, y27)
            hip_y = (x28, y28)
            calc_shoulder_width = distance.euclidean(hip_x, hip_y)

            scale = real_shoulder_width / calc_shoulder_width
            # print(scale)
            scale = 0.005

            for mark in range(33):
                # selected = "world"
                selected = "not_world"
                if selected == "world":
                    x = results.pose_world_landmarks.landmark[mark].x
                    y = results.pose_world_landmarks.landmark[mark].y
                    z = results.pose_world_landmarks.landmark[mark].z

                    deg45 = np.deg2rad(-25)
                    cos = np.cos(deg45)
                    sin = np.sin(deg45)
                    rot_y = (y * cos) - (z * sin)
                    rot_z = (y * sin) + (z * cos)

                    pos[mark][0] = x
                    pos[mark][2] = -rot_y
                    pos[mark][1] = rot_z

                else:
                    x = results.pose_landmarks.landmark[mark].x
                    y = results.pose_landmarks.landmark[mark].y
                    z = results.pose_landmarks.landmark[mark].z

                    deg45 = np.deg2rad(-25)
                    cos = np.cos(deg45)
                    sin = np.sin(deg45)
                    rot_y = (y * cos) - (z * sin)
                    rot_z = (y * sin) + (z * cos)

                    offset_z = -0.2
                    pos[mark][0] = (x - 0.5) * cap_width * scale
                    pos[mark][2] = (1 - rot_y) * cap_height * scale
                    pos[mark][1] = -rot_z + offset_z

                    # pos[mark][0] = (x - 0.5) * cap_width * scale
                    # pos[mark][2] = (1 - y) * cap_height * scale
                    # pos[mark][1] = z

                pos_str = ""
                pos_str += str(time.time()) + ","
                for pos_one in pos:
                    for pos_one_one in pos_one:
                        pos_str += str(pos_one_one) + ","
                # pose_logger.on_message(pos_str)

                with open("pos.pkl", "r+b") as f:
                    mm = mmap.mmap(f.fileno(), 0)
                    pos_list_pkl = pickle.dump(pos, mm)

            # shlder
            mark_list = [27, 28]
            for mark in mark_list:
                x = pos[mark][0]
                y = pos[mark][2]
                z = pos[mark][1]

                enable = 1
                timeoffset = 0

                if mark == 27:
                    index = 1
                elif mark == 28:
                    index = 2
                else:
                    pass

                osc_client_inst.send_room_unity(
                    index,
                    enable,
                    timeoffset,
                    x=x,
                    y=y,
                    z=z,
                    qx=1.0,
                    qy=0,
                    qz=0,
                    qw=0,
                )

            # hip
            x = (pos[23][0] + pos[24][0]) / 2
            y = (pos[23][2] + pos[24][2]) / 2
            z = (pos[23][1] + pos[24][1]) / 2

            enable = 1
            timeoffset = 0

            index = 0
            # print(offset)
            osc_client_inst.send_room_unity(
                index,
                enable,
                timeoffset,
                x=x,
                y=y,
                z=z,
                qx=1.0,
                qy=0,
                qz=0,
                qw=0,
            )
            # osc_client_inst.send_joint_unity()
            # osc_client_inst.update()

        end_time = time.time()
        diff_time = end_time - start_time
        # print(f"process time is {diff_time}")

        cv2.imshow("holistic detection", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

camera.release()
