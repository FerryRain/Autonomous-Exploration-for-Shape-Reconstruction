import time

import numpy as np

from utils.Controller import XArm_Controller
from utils.Ft_Sensors import DR304ForceSensor


class ForceDrivenCompliantController:
    def __init__(self, ip, port="COM3", dt=0.02, alpha=0.2):
        self.dt = dt
        self.alpha = alpha
        self.f_target = 1.5
        self.k_f = 1e4
        self.Kd = np.array([0.1, 0.1, 0.1])
        self.Ka = np.array([0.01, 0.01, 0.01])
        self.v_min = 1.0
        self.v_max = 100.0

        self.arm = XArm_Controller(ip)
        self.sensor = DR304ForceSensor(port=port)

        self.prev_pos = None
        self.prev_vel = np.zeros(3)
        self.prev_acc = np.zeros(3)

    def estimate_derivatives(self, curr_pos):
        if self.prev_pos is None:
            self.prev_pos = curr_pos
            return np.zeros(3), np.zeros(3)

        raw_vel = (curr_pos - self.prev_pos) / self.dt
        vel = self.alpha * raw_vel + (1 - self.alpha) * self.prev_vel
        raw_acc = (vel - self.prev_vel) / self.dt
        acc = self.alpha * raw_acc + (1 - self.alpha) * self.prev_acc

        self.prev_pos = curr_pos
        self.prev_vel = vel
        self.prev_acc = acc

        return vel, acc

    def read_contact_force(self):
        return np.array(self.sensor.read_force_data()[:3])

    def step(self):
        curr_pose = self.arm.get_ee_position()
        curr_pos = curr_pose[:3]

        vel, acc = self.estimate_derivatives(curr_pos)
        f_contact = self.read_contact_force()
        norm_f = np.linalg.norm(f_contact)
        if norm_f < 3e-1:
            return
        normal = f_contact / norm_f

        force_term = self.k_f * (self.f_target - norm_f) * -normal

        damping_term = -self.Kd * vel
        inertia_term = -self.Ka * acc

        u = force_term + damping_term + inertia_term
        delta_pos = u * self.dt

        delta_pose = np.concatenate([delta_pos, np.zeros(3)])
        speed = np.clip(np.linalg.norm(delta_pos) / self.dt, self.v_min, self.v_max)
        self.arm.move_to(delta_pose, is_delta=True, speed=speed)

    def run(self):
        try:
            while True:
                self.step()
                # time.sleep(self.dt)
        except KeyboardInterrupt:
            self.arm.stop()
            print("[Stopped] Force-driven compliant controller stopped.")


class Real_Robot:
    def __init__(self, ip, port="COM3", f_target: float = 1.5, k_f: float = 0.001, dt: float = 0.01,
                       v_min=1, speed_scale=50, v_max=100):
        self.arm = XArm_Controller(ip)
        self.Ft_Sensor = DR304ForceSensor(port=port)
        self.f_target = f_target
        self.k_f = k_f
        self.dt = dt
        self.v_min = v_min
        self.v_max = v_max
        self.speed_scale = speed_scale

    def force_feedback_control(self, normal, f_contact):
        u, v = self.force_feedback(normal, f_contact)

        self.arm.move_to(u, is_delta=True, speed=v)
        # print(np.linalg.norm(u[:3]))
        # return np.linalg.norm(u[:3])
        return u[:3]

    def force_feedback(self, normal, f_contact):
        f_contact_n = np.dot(f_contact, normal)
        f_err = self.f_target - f_contact_n
        f_err_n = self.k_f * f_err * -normal
        delta_pos = f_err_n * self.dt

        norm = np.linalg.norm(delta_pos)
        delta_pos = delta_pos if norm <= 4 else delta_pos / norm * 4
        delta_pose = np.concatenate([delta_pos, np.zeros(3)])

        move_speed = self.v_min + abs(np.linalg.norm(f_err) + 1e-10) * self.speed_scale
        move_speed = min(max(move_speed, self.v_min), self.v_max)

        return delta_pose, move_speed