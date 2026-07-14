import time
import numpy as np
import cv2
import keyboard
from utils.Attaching_Controller import Real_Robot

robot = Real_Robot(
    ip="10.19.131.200",
    port="COM3",
    f_target=4.0,
    k_f=6e3,
    dt=0.0001,
    v_min=1,
    speed_scale=60,
    v_max=100
)

contact_force = []
displacement_direction = []

camera_id = 0
cap = cv2.VideoCapture(camera_id)
if not cap.isOpened():
    raise IOError("Falied loading cam")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter("../Video_output/ACC/video_OUT.mp4", fourcc, fps, (width, height))

print("Start running(press q to quit)")
state = False
origin = (width // 2, height // 2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falied loading cam")
        break

    f_contact = np.array(robot.Ft_Sensor.read_force_data()[:3])
    norm_f_contact = np.linalg.norm(f_contact + 1e-10)
    normal = f_contact / norm_f_contact

    if norm_f_contact > 0.5:
        state = True
        contact_force.append(f_contact.copy())
        if norm_f_contact > 5:
            delta = - (normal * robot.force_feedback_control(normal, f_contact))
        else:
            delta = normal * robot.force_feedback_control(normal, f_contact)
        displacement_direction.append(delta.copy())

        proj_dir = np.array([delta[1], delta[2]])
        norm = np.linalg.norm(proj_dir)
        if norm > 1e-8:
            proj_dir /= norm
            arrow_len = int(100 * norm_f_contact)
            end_point = (
                int(origin[0] - arrow_len * proj_dir[0]),
                int(origin[1] - arrow_len * proj_dir[1])
            )
            cv2.arrowedLine(frame, origin, end_point, (0, 0, 255), 3, tipLength=0.2)
        cv2.putText(frame, f"Force: {norm_f_contact:.2f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif state:
        contact_force.append(np.array([0.0, 0.0, 0.0]))
        displacement_direction.append(np.array([0.0, 0.0, 0.0]))

    out_video.write(frame)
    if keyboard.is_pressed('q'):
        print("Detected 'q'. 保存数据中...")
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()

np.savetxt("../Video_output/ACC/contact_force.csv", np.vstack(contact_force), delimiter=",")
np.savetxt("../Video_output/ACC/displacement_direction.csv", np.vstack(displacement_direction), delimiter=",")

print("Complete：video.mp4, contact_force.csv, displacement_direction.csv")
