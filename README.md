# AMMH / AESRM

## Repository Structure

```text
.
|-- Env/
|   |-- Real/
|   |   |-- Real_exp_1/              # Real-world experiments without global exploration
|   |   |-- Real_exp_2(global)/      # Real-world experiments with global exploration
|   |   |-- Real_exp_3(ER_GPIS)/     # Real-world experiments using ER-GPIS
|   |   |-- AblationStudy/           # Ablation studies and comparative experiments
|   |   |-- Outer_shape*/            # Outer-contour exploration experiments
|   |   `-- video_process.py         # Video and result post-processing
|   |-- Sim/                         # Simulation experiments
|   `-- Results/                     # Experimental outputs
|-- utils/
|   |-- HE_GPIS.py                   # GPIS, HE-GPIS, uncertainty-aware GPIS, and ER-GPIS classes
|   |-- utils.py                     # Utilities for GPIS initialization, contact-point storage, path generation, etc.
|   |-- Attaching_Controller.py      # Real_Robot wrapper and force-feedback controller
|   |-- Controller.py                # xArm controller wrapper
|   `-- Ft_Sensors.py                # DR304 force-sensor interface
|-- environment.yaml                 # Conda environment configuration
`-- cg_baseline.py                   # Baseline script
```

## Environment Setup

Create the Conda environment using the configuration file provided in the repository:

```bash
conda env create -f environment.yaml
conda activate AMMH
```

Before running real-world experiments, verify the following:

- The xArm is connected, and the host PC can access the robot IP specified in the script, for example, `10.19.131.200`.
- The DR304 force sensor is connected, and its serial port matches the value specified in the script, such as `COM3` or `COM4`.
- The code explicitly uses `cuda` or `cuda:0` in several places, so a CUDA-enabled installation of PyTorch is required.
- The robot's initial pose, workspace, probe configuration, and object placement have been manually checked before execution.

## Running the Three Types of Real-World Experiments

The real-world experiments are divided into three main configurations:

| Mode | Directory | Description |
| --- | --- | --- |
| Real_exp1 | `Env/Real/Real_exp_1` | Version without global exploration. It mainly performs local contact-guided exploration and periodically updates the GPIS model. |
| Real_exp2 | `Env/Real/Real_exp_2(global)` | Version with global exploration. It is used as a comparison setting that includes a global exploration stage. |
| Real_exp3 | `Env/Real/Real_exp_3(ER_GPIS)` | ER-GPIS version. It uses `init_normal_HER_GPIS_uncertainty(...)` and introduces uncertainty-guided global motions during exploration. |

Each directory contains scripts for different object geometries, for example:

- `Real_exp(irregular).py`
- `Real_exp(ellipse).py`
- `Real_exp(d_s).py`
- Sphere-related scripts

Example commands executed from the repository root:

```bash
python "Env/Real/Real_exp_1/Real_exp(irregular).py"
python "Env/Real/Real_exp_2(global)/Real_exp(irregular).py"
python "Env/Real/Real_exp_3(ER_GPIS)/Real_exp(irregular).py"
```

Before execution, update the hardware parameters near the end of the selected script:

```python
real_robot = Real_Robot(
    ip="10.19.131.200",
    port="COM3",
    f_target=1.0,
    k_f=1e3,
    v_min=1,
    dt=0.0001,
    speed_scale=60,
    v_max=100,
)
```

## Parameters for Objects of Different Sizes

For objects with different dimensions, the main parameters to adjust are the GPIS template bounds and the z-axis limits used for generating motion points.

```python
gpis_temp_x = 25
gpis_temp_y = 25
gpis_temp_z = 10
```

These parameters define the GPIS query and reconstruction region:

```python
temp_min = np.array([-gpis_temp_x, -gpis_temp_y, -gpis_temp_z])
temp_max = np.array([ gpis_temp_x,  gpis_temp_y,  gpis_temp_z])
```

If the object is wider or taller, or if the initial contact point is farther from the object center, increase the corresponding `gpis_temp_*` value. The complete object surface and all potential contact regions should remain within the GPIS bounds.

```python
generate_z_min = -4
generate_z_max = 4
```

These two parameters constrain the generated motion range along the z-axis:

- For tall objects, appropriately increase the range defined by `generate_z_min` and `generate_z_max`.
- For flat objects, reduce the z-axis range to avoid unnecessary or potentially unsafe vertical motion.
- Before running a real-world experiment, verify the safety limits against both the robot workspace and the object height.

The current real-world experiment scripts use utility functions ending in `*_mm`. Parameter units must therefore remain consistent with the robot displacement units used by the selected script.

## GPIS Classes and Utility Functions

The GPIS-related classes are implemented in `utils/HE_GPIS.py`:

- `GPIS`
- `global_HE_GPIS`
- `local_HE_GPIS`
- `normal_HE_GPIS`
- `HE_GPIS_Uncertainty_draw`
- `HER_GPIS_Uncertainty_draw`
- `HER_GPIS_Uncertainty_wo_dummy`

The experiment scripts usually create these models through initialization functions in `utils/utils.py` instead of instantiating the classes directly:

```python
gpis = init_normal_HE_GPIS_uncertainty(...)
er_gpis = init_normal_HER_GPIS_uncertainty(...)
```

Frequently used utility functions for real-world experiments are also defined in `utils/utils.py`:

- `generate_next_point_limited_z_force_normal_mm(...)`
- `store_contact_points_real_normals_mm(...)`
- `update_gpis_uncertainty_mm(...)`
- `store_global_points(...)`

## Output Results

Each run saves the reconstructed GPIS surface, exploration points, and global buffer according to the `gpis_store_path` specified in the script. For example, the ER-GPIS irregular-object script uses:

```python
gpis_store_path = "../../Results/RealExp_3/Real_exp_3_irregular_"
```

The GPIS class appends a timestamp to this prefix and saves the resulting `.npy` files in the corresponding directory.

## Real-World Experiment Checklist

Before running a real-robot script, verify the following items:

1. The robot IP address and force-sensor serial port are correct.
2. The object lies entirely within the GPIS template bounds.
3. `generate_z_min` and `generate_z_max` do not allow the robot to leave its safe workspace.
4. `force_threshold`, `f_target`, `k_f`, `dt`, and the velocity limits are appropriate for the current probe and object.
5. The initial contact and the `init_gpis(...)` initialization stage are manually supervised.

## Citation

If this project is useful for your research, please cite:

```bibtex
@article{zhao2025aesrm,
  title   = {Autonomous Exploration for Shape Reconstruction and Measurement via Informative Contact-Guided Planning},
  author  = {Zhao, Feiyu and Xiao, Chenxi},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2025},
  doi     = {10.1109/LRA.2025.3641100}
}
```