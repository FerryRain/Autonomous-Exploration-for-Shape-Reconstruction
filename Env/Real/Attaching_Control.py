import time

import numpy as np
import keyboard
from utils.Attaching_Controller import Real_Robot


if __name__ == '__main__':

    robot = Real_Robot("10.19.131.200", port="COM3", f_target=1.,
                                 k_f=6e3,
                                 v_min=1,
                                 dt=0.0001,
                                 speed_scale=60,
                                 v_max=100
                                 )
    contact_force = []
    displacement_direction = []
    displacement_distance = []
    state = False
    while True:
        f_contact = np.array(robot.Ft_Sensor.read_force_data()[:3])
        # f_contact[0] = f_contact[0]
        norm_f_contact = np.linalg.norm(f_contact + 1e-10)
        normal = f_contact / norm_f_contact

        if norm_f_contact > 2e-1:
            state = True
            contact_force.append(norm_f_contact)
            displacement_direction.append(normal * robot.force_feedback_control(normal, f_contact))
        elif state == True:
            contact_force.append(np.float64(0))
            displacement_direction.append(np.array([0.0, 0.0, 0.0]))

