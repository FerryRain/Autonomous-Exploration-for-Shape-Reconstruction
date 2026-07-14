"""
@FileName：Real_Test_2（Attaching_Control）.py
@Description：
@Author：Ferry
@Time：2025 5/29/25 12:59 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""

from collections import deque

# from utils.Attaching_Controller import Real_Robot
from utils.utils import *


def run_exploration(robot: Real_Robot):
    count = 0

    arm = robot.arm
    sensor = robot.Ft_Sensor
    init_position = arm.init_ee_pos

    origin_pos = init_position[:3]
    origin_pos[2] = origin_pos[2]
    origin_rot = init_position[3:]
    pos = torch.tensor([0, .8, 1])
    rot = torch.zeros(3, dtype=torch.float)
    target_pos, target_quat = pos.reshape(-1), rot.reshape(-1)
    next_position = torch.cat([target_pos, target_quat], dim=0)
    arm.move_to(next_position.numpy())

    # Initialize buffer
    touched_buf = torch.empty((0, 3), device="cuda")
    untouched_buf = torch.empty((0, 3), device="cuda")
    explored_queue = deque(maxlen=10)

    gpis_res = 50
    gpis_grid_count = 4
    gpis_store_path = "../../Results/ablation/Diress/global/Diress_global_"

    gpis_temp_x = 25
    gpis_temp_y = 25
    gpis_temp_z = 10
    #
    # gpis_temp_bb_x = .020
    # gpis_temp_bb_y = .020
    # gpis_temp_bb_z = .0075

    # gpis_temp_x = 28
    # gpis_temp_y = 28
    # gpis_temp_z = 13
    #
    # gpis_temp_bb_x = 26
    # gpis_temp_bb_y = 26
    # gpis_temp_bb_z = 10

    distance_tol = 2e-2
    force_threshold = 3e-1

    generate_alpha = -0.4
    generate_beta = 0.1
    # generate_alpha = -0.4/10
    # generate_beta = 0.88/30
    generate_z_min = -4.2
    generate_z_max = 4.2
    generate_lambda_penalty = 1.0
    generate_gamma = 500
    # z 522.5 521.5 512.5
    # y 409.5 408.5 383.5
    # x

    temp_min = np.array([-gpis_temp_x, -gpis_temp_y, -gpis_temp_z])
    temp_max = np.array([gpis_temp_x, gpis_temp_y, gpis_temp_z])

    # temp_min = np.array([-gpis_temp_x, -gpis_temp_y, -gpis_temp_z])
    # temp_max = np.array([gpis_temp_x, gpis_temp_y, gpis_temp_z])

    points_num = 3
    intervals = 0.5

    gpis = init_normal_HE_GPIS_model(temp_min, temp_max, res=gpis_res,
                                           grid_count=gpis_grid_count,
                                           store_path=gpis_store_path, training_iter=50)

    # init_contact
    while True:
        f = np.array(sensor.read_force_data()[:3])
        f_norm = np.linalg.norm(np.array(sensor.read_force_data()[:3]))
        if f_norm > force_threshold:
            break
        robot.arm.move_to(np.array([0.3, 0, 0, 0, 0, 0]), is_delta=True, speed=10)

    while True:
        touched_buf, untouched_buf = store_contact_points_real(sensor, arm, origin_pos.copy(),
                                                                                       touched_buf,
                                                                                       untouched_buf,
                                                                                       threshold=force_threshold,
                                                                                       tol=distance_tol)
        if (arm.get_ee_position()[:3] - origin_pos)[-1] > generate_z_max:
            break
        robot.arm.move_to(np.array([0, 0, 0.3, 0, 0, 0]), is_delta=True, speed=10)

    gpis, _ = update_gpis(arm, explored_queue, touched_buf, untouched_buf, gpis)
    touched_buf = torch.empty((0, 3), device="cuda")
    untouched_buf = torch.empty((0, 3), device="cuda")
    count = 0

    # Simulate physics
    while gpis.time_step < 100:
        if get_exploration_status_real(count, 25):
            gpis, _ = update_gpis(arm, explored_queue, touched_buf, untouched_buf, gpis)
            touched_buf = torch.empty((0, 3), device="cuda")
            untouched_buf = torch.empty((0, 3), device="cuda")
            count = 0


        # tb changed
        # generate_next_point_limited_z_force_normal_mm(robot, arm, sensor, origin_pos.copy(), gpis,
        #                                            alpha_base=generate_alpha, beta_base=generate_beta,
        #                                            z_min=generate_z_min, z_max=generate_z_max,
        #                                            lambda_penalty=generate_lambda_penalty, gamma=generate_gamma)

        f = generate_next_point_limited_z_force_normal_mm_outer(robot, arm, sensor, origin_pos.copy(), gpis,
                                                   alpha_base=generate_alpha, beta_base=generate_beta,
                                                   z_min=generate_z_min, z_max=generate_z_max,
                                                   lambda_penalty=generate_lambda_penalty, gamma=generate_gamma, f_buf=f)

        if count % 1 == 0:
            touched_buf, untouched_buf = store_contact_points_real(sensor, arm, origin_pos.copy(),
                                                                                touched_buf,
                                                                                untouched_buf,
                                                                                threshold=force_threshold,
                                                                                tol=distance_tol)

        if gpis.time_step == 50  or gpis.time_step == 70:
            direction_xy = gpis.get_max_uncertainty_xy_direction(arm.get_ee_position()[:3] - origin_pos, z_min=-3, z_max=3)
            step = 0
            print("Global exploration")
            while True:
                f = np.array(sensor.read_force_data()[:3])
                f_norm = np.linalg.norm(f)
                if f_norm > force_threshold and step >= 5:
                    print("Back to local exploration")
                    gpis.time_step += 1
                    break
                target = direction_xy
                target = torch.tensor(target, dtype=torch.float32, device='cpu').reshape(-1)
                rot = torch.zeros(3, dtype=torch.float)
                target, quat = target.reshape(-1), rot.reshape(-1)
                target = (target / target.norm(p=2)) * 0.3
                next_position = torch.cat([target, quat], dim=0)
                arm.move_to(next_position.numpy())
                step += 1
        if gpis.time_step % 10 == 0 and gpis.time_step >= 70:
            direction_xy = gpis.get_max_uncertainty_xy_direction(arm.get_ee_position()[:3] - origin_pos, z_min=-3,
                                                                 z_max=3)
            step = 0
            print("Global exploration")
            while True:
                f = np.array(sensor.read_force_data()[:3])
                f_norm = np.linalg.norm(f)
                if f_norm > force_threshold and step >= 5:
                    print("Back to local exploration")
                    gpis.time_step += 1
                    break
                target = direction_xy
                target = torch.tensor(target, dtype=torch.float32, device='cpu').reshape(-1)
                rot = torch.zeros(3, dtype=torch.float)
                target, quat = target.reshape(-1), rot.reshape(-1)
                target = (target / target.norm(p=2)) * 0.3
                next_position = torch.cat([target, quat], dim=0)
                arm.move_to(next_position.numpy())
                step += 1

        # time +1
        count += 1


if __name__ == '__main__':
    real_robot = Real_Robot(ip="10.19.131.200", port="COM3", f_target=1.0,
                            k_f=1e3,
                            v_min=1,
                            dt=0.0001,
                            speed_scale=60,
                            v_max=100)

    # Run the simulator
    run_exploration(real_robot)
