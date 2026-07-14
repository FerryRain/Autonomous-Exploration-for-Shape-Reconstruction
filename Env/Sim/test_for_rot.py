"""
@FileName：test_for_rot.py
@Description：
@Author：Ferry
@Time：2025 5/26/25 2:55 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""
"""
@FileName：Exploration_env_stage7_1(sphere).py
@Description：
@Author：Ferry
@Time：2025 5/19/25 3:11 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""


import argparse

from isaaclab.app import AppLauncher
from utils.Controller import Pid_Controller
from utils.utils import *

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different dexterous hands.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from collections import deque


@configclass
class ExpllorationEnvCfg(InteractiveSceneCfg):
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot = RigidObjectCollectionCfg(
        rigid_objects={"dot": RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.SphereCfg(
                radius=0.05,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, ),
                mass_props=sim_utils.MassPropertiesCfg(mass=5),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.0, .0, 1.0)),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.16)),
        )
        }
    )

"""
--------------
# move to func
--------------
"""


def move_to_base(robot, controller, pos, quat):

    force, torques = controller.step(robot.data.object_state_w.reshape(-1, 13))
    robot.set_external_force_and_torque(forces=force.reshape(-1, 1, 3).to("cuda:0"),
                                        torques=torques.reshape(-1, 1, 3).to("cuda:0"),
                                        object_ids=[0])







def run_simulator(sim: sim_utils.SimulationContext, entities):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Define the frame marker
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (.05, .05, .05)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Init Robot state
    robot = entities['robot']
    pos = torch.tensor([0, 0., 0.0], device="cuda")
    quat = torch.tensor((0.257551, 0.283045, 0.683330, -0.621782), device="cuda")
    target_pos, target_quat = pos.reshape(-1, 3) + entities.env_origins[0], quat.reshape(-1, 4)

    K_p_pos, K_i_pos, K_d_pos = 1.25, 0.00001, 1.4  # pos p i d of  PID
    K_p_rot, K_i_rot, K_d_rot = 0.1, 0.0, 0.1  # rot p i d of  PID
    dt = 0.01
    controller = Pid_Controller(K_p_pos, K_i_pos, K_d_pos, K_p_rot, K_i_rot, K_d_rot, 1, dt, local_control=True)
    controller.reset(torch.cat((target_pos, target_quat), dim=-1).reshape(-1, 7), [0])

    estimated_surface = []
    # Simulate physics
    while simulation_app.is_running():
        if count == 1000:
            d_Normal = torch.tensor([0., 0., 1.], device="cuda").reshape(-1,3)
            q_target = quat_align_z_to_vector(d_Normal).reshape(-1, 4)
            target_quat = check_quat_validity(q_target)
            controller.reset(torch.cat((target_pos, target_quat), dim=-1).reshape(-1, 7), [0])
        if count == 2000:
            d_Normal = torch.tensor([0., 1., 0.], device="cuda").reshape(-1,3)
            q_target = quat_align_z_to_vector(d_Normal).reshape(-1, 4)
            target_quat = check_quat_validity(q_target)
            controller.reset(torch.cat((target_pos, target_quat), dim=-1).reshape(-1, 7), [0])
        if count == 3000:
            d_Normal = torch.tensor([1., 0., 0.], device="cuda").reshape(-1,3)
            q_target = quat_align_z_to_vector(d_Normal).reshape(-1, 4)
            target_quat = check_quat_validity(q_target)
            controller.reset(torch.cat((target_pos, target_quat), dim=-1).reshape(-1, 7), [0])
        move_to_base(robot, controller, target_pos, target_quat)

        # write data to sim
        ee_marker.visualize(robot.data.object_pos_w[:, 0, :].reshape(-1, 3) + entities.env_origins[0],
                            robot.data.object_quat_w[:, 0, :].reshape(-1, 4))
        goal_marker.visualize(target_pos + entities.env_origins[0],
                              target_quat)
        robot.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


if __name__ == '__main__':
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.0, -0.5, 1.5], target=[0.0, -0.2, 0.5])
    # design scene
    scene_cfg = ExpllorationEnvCfg(num_envs=args_cli.num_envs, env_spacing=2.0)

    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Run the simulator
    run_simulator(sim, scene)
