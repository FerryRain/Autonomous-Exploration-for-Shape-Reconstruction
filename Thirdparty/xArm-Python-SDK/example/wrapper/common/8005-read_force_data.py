#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2020, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
Description: read force raw data and external data
"""

import sys
import time
import math
import socket
import struct

import os

from xarm.wrapper import XArmAPI
#######################################################

if len(sys.argv) >= 2:
    ip = sys.argv[1]
else:
    try:
        from configparser import ConfigParser
        parser = ConfigParser()
        parser.read('../robot.conf')
        ip = parser.get('xArm', 'ip')
    except:
        ip = input('Please input the xArm ip address:')
        if not ip:
            print('input error, exit')
            sys.exit(1)
########################################################

arm = XArmAPI(ip, enable_report=True)
arm.motion_enable(enable=True)
arm.ft_sensor_enable(0)

arm.clean_error()
arm.clean_warn()
arm.ft_sensor_enable(1)
time.sleep(0.5)
arm.ft_sensor_set_zero()


import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


while arm.connected and arm.error_code == 0:
    # ft_raw_force and ft_ext_force will update by reporting socket
    # print('raw_force: {}'.format(arm.ft_raw_force))
    print('exe_force: {}'.format(arm.ft_ext_force))

    # # get_ft_sensor_data() will get the last ext_force
    # code, ext_force = arm.get_ft_sensor_data()
    # if code == 0:
    #     print('exe_force: {}'.format(ext_force))






    # 调用清屏函数

    time.sleep(0.05)
    clear_console()

arm.ft_sensor_enable(0)
arm.disconnect()
