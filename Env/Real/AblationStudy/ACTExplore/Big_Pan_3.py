"""
@FileName：Real_Test_2（Attaching_Control）.py
@Description：
@Author：Ferry
@Time：2025 5/29/25 12:59 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""
import os
import time
from collections import deque

# from utils.Attaching_Controller import Real_Robot
from utils.utils import *
#632 597.8

def run_exploration(robot: Real_Robot):
    count = 0

    arm = robot.arm
    sensor = robot.Ft_Sensor
    init_position = arm.init_ee_pos

    origin_pos = init_position[:3]
    origin_pos[2] = origin_pos[2] + 17.5

    pos = torch.tensor([0, .8, 1])
    rot = torch.zeros(3, dtype=torch.float)
    target_pos, target_quat = pos.reshape(-1), rot.reshape(-1)
    next_position = torch.cat([target_pos, target_quat], dim=0)
    arm.move_to(next_position.numpy())

    # Initialize buffer
    touched_buf = torch.empty((0, 3), device="cuda")
    untouched_buf = torch.empty((0, 3), device="cuda")
    global_buffer = torch.empty((0, 3), device="cuda")
    explored_queue = deque(maxlen=10)

    gpis_res = 100
    gpis_grid_count = 6
    gpis_store_path = "../../../Results/ablation/ACTExplore/Big_pan/Big_pan_"

    gpis_temp_x = 100
    gpis_temp_y = 100
    gpis_temp_z = 50

    # distance_tol = 1e-2
    distance_tol = 1e0
    force_threshold = 5e-1

    generate_alpha = 0.4
    generate_beta = 0.1
    generate_z_min = -17.5
    generate_z_max = 17
    generate_lambda_penalty = 1.0
    generate_gamma = 500

    temp_min = np.array([-gpis_temp_x, -gpis_temp_y, -gpis_temp_z])
    temp_max = np.array([gpis_temp_x, gpis_temp_y, gpis_temp_z])

    points_num = 2
    intervals = 5

    gpis = init_normal_HER_GPIS_uncertainty(temp_min, temp_max, points_num, intervals, res=gpis_res,
                                           grid_count=gpis_grid_count,
                                           store_path=gpis_store_path, training_iter=20)

    for i in range(260):
        pos = torch.tensor([-0.3, 0, 0])
        rot = torch.zeros(3, dtype=torch.float)
        target_pos, target_quat = pos.reshape(-1), rot.reshape(-1)
        next_position = torch.cat([target_pos, target_quat], dim=0)
        arm.move_to(next_position.numpy())
        time.sleep(0.05)
        # touched_buf, untouched_buf = store_contact_points_real(sensor, arm, origin_pos.copy(), touched_buf,
        #                                                        untouched_buf, threshold=5e-1)
    time.sleep(0.5)
    # init_contact
    while True:
        f = np.array(sensor.read_force_data()[:3])
        f_norm = np.linalg.norm(np.array(sensor.read_force_data()[:3]))
        if f_norm > force_threshold:
            break
        robot.arm.move_to(np.array([0.0, 0, 0.5, 0, 0, 0]), is_delta=True, speed=10)

    gpis.init_gpis(f / f_norm, torch.tensor(arm.get_ee_position()[:3] - origin_pos))

    normals_buf = torch.empty((0, 3), device="cuda")

    # Simulate physics
    while True:
        if get_exploration_status_real(count, 25):
            gpis = update_gpis_uncertainty_mm(arm, explored_queue, touched_buf, untouched_buf, gpis, normals_buf)
            touched_buf = torch.empty((0, 3), device="cuda")
            untouched_buf = torch.empty((0, 3), device="cuda")
            normals_buf = torch.empty((0, 3), device="cuda")
            torch.cuda.empty_cache()
            count = 0

        # generate_next_point_limited_z_force_normal_mm(robot, arm, sensor, origin_pos.copy(), gpis,
        #                                            alpha_base=generate_alpha, beta_base=generate_beta,
        #                                            z_min=generate_z_min, z_max=generate_z_max,
        #                                            lambda_penalty=generate_lambda_penalty, gamma=generate_gamma)
        # star = time.time()
        f = generate_next_point_limited_z_force_normal_mm_outer(robot, arm, sensor, origin_pos.copy(), gpis,
                                                   alpha_base=generate_alpha, beta_base=generate_beta,
                                                   z_min=generate_z_min, z_max=generate_z_max,
                                                   lambda_penalty=generate_lambda_penalty, gamma=generate_gamma, f_buf=f, threshold=force_threshold)
        # end = time.time() - star

        if count % 1 == 0:
            touched_buf, untouched_buf, normals_buf = store_contact_points_real_normals_mm(sensor, arm, origin_pos.copy(),
                                                                                        touched_buf,
                                                                                        untouched_buf, normals_buf,
                                                                                        threshold=force_threshold,
                                                                                        tol=distance_tol)
        # time +1
        count += 1

        # if gpis.time_step != 0 and gpis.time_step % 50 == 0:
        #     direction_xy = gpis.get_max_uncertainty_xy_direction(arm.get_ee_position()[:3] - origin_pos, z_min=-3,
        #                                                          z_max=3)
        #     target = direction_xy
        #     target = torch.tensor(target, dtype=torch.float32, device='cpu').reshape(-1)
        #     rot = torch.zeros(3, dtype=torch.float)
        #     target, quat = target.reshape(-1), rot.reshape(-1)
        #     target = (target / target.norm(p=2)) * 0.3
        #
        #     target = torch.cat([target, quat], dim=0)
        #     print("Global exploration")
        #
        #     for i in range(50):
        #         pos = torch.tensor([0, 0, -0.5])
        #         rot = torch.zeros(3, dtype=torch.float)
        #         target_pos, target_quat = pos.reshape(-1), rot.reshape(-1)
        #         next_position = torch.cat([target_pos, target_quat], dim=0)
        #         arm.move_to(next_position.numpy())
        #         time.sleep(0.05)
        #         global_buffer = store_global_points(arm, origin_pos.copy(),
        #                                             global_buffer,
        #                                             tol=1.0)
        #
        #     arm.arm.set_mode(0)
        #     arm.arm.set_state(0)
        #     arm.arm.set_servo_angle(angle=arm.init_joint_pos, speed=2, mvacc=200, wait=True)
        #
        #     time.sleep(0.5)
        #
        #     for i in range(100):
        #         arm.move_to(target.numpy())
        #         time.sleep(0.05)
        #         global_buffer = store_global_points(arm, origin_pos.copy(),
        #                                             global_buffer,
        #                                             tol=1.0)
        #
        #     while True:
        #         f = np.array(sensor.read_force_data()[:3])
        #         f_norm = np.linalg.norm(np.array(sensor.read_force_data()[:3]))
        #         if f_norm > force_threshold:
        #             break
        #         robot.arm.move_to(np.array([0.0, 0, 0.3, 0, 0, 0]), is_delta=True, speed=10)
        #         global_buffer = store_global_points(arm, origin_pos.copy(),
        #                                             global_buffer,
        #                                             tol=1.0)
        #
        #
        #     save_path = os.path.join(gpis.store_path, f"global_buffer_{gpis.time_step}.npy")
        #     np.save(save_path, global_buffer.cpu().numpy())
        #     gpis.time_step += 1
        #
        #     print("--------------------------------------------------------")
        #     print(f"saved explored_x_{gpis.time_step}.npy to", save_path)

        if gpis.time_step % 10 == 0:
            gpis.down_sample_points(r=4.0)
            gpis.init_gpis(f, torch.tensor(arm.get_ee_position()[:3] - origin_pos))


if __name__ == '__main__':
    real_robot = Real_Robot(ip="10.19.131.200", port="COM4", f_target=1.0,
                            k_f=1e3,
                            v_min=1,
                            dt=0.0001,
                            speed_scale=60,
                            v_max=100)

    # Run the simulator
    run_exploration(real_robot)
