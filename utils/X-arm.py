
import time

import numpy as np
import torch
from xarm.wrapper import XArmAPI

class XArm_Controller:
    def __init__(self, ip):
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.init_joint_pos = self.arm.get_initial_point()[1]
        self.initial_to_initial_pos()
        self.next_pos = self.init_ee_pos

    def initial_to_initial_pos(self):
        self.arm.set_state(0)
        self.arm.set_servo_angle(angle=self.init_joint_pos, speed=20, wait=False)
        print("-----------------------------------------------")
        print("         Please wait about 10 seconds")
        time.sleep(3)
        print("           Success initial position")
        print("-----------------------------------------------")
        self.init_ee_pos = np.array(self.get_ee_position())

    def ee_back_to_initial_pos(self):
        self.arm.set_state(0)
        self.arm.set_position(x=self.init_ee_pos[0], y=self.init_ee_pos[1], z=self.init_ee_pos[2], roll=self.init_ee_pos[3],
                              pitch=self.init_ee_pos[4], yaw=self.init_ee_pos[5],
                              speed=20, is_radian=False, wait=False)
        print("-----------------------------------------------")
        print("    Success backed to the initial position")
        print("-----------------------------------------------")

    def get_ee_position(self):
        return np.array(self.arm.get_position(is_radian=False)[1])

    def move_to(self, pos, is_delta=True, speed=20):
        if not is_delta:
            self.next_pos = pos
        else:
            self.next_pos = self.get_ee_position() + pos
        self.step(speed=speed)

    def step(self, speed=1, mvacc=1):
        self.arm.set_state(0)
        self.arm.set_mode(1)
        self.arm.set_servo_cartesian(self.next_pos.tolist(), speed=speed, mvacc=mvacc)  # , mvacc=2000

    def stop(self):
        self.arm.emergency_stop()

if __name__ == '__main__':
    arm = XArm_Controller("10.19.131.200")
    # arm.move_to(pos=[0,100,-100,0,0,0])
    from Ft_Sensors import DR304ForceSensor

    sensor = DR304ForceSensor(port="COM3")
    arm.move_to(pos=[1, 0, 0, 0, 0, 0], speed=1)
    while True:
        arm.move_to(pos=[1, 0, 0, 0, 0, 0], speed=1)
        if sensor.get_contact():
            arm.stop()
            break