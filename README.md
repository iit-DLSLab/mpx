<p align="left">
 <img src="https://github.com/user-attachments/assets/faaee057-131d-47da-b841-8832d536e5c5" width="70%" /> 
</p>

This repo implements the code for legged robot MPC and Trajectory Optimization all in JAX. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/de8b9650-684e-4f31-82e4-9a0035f50f8e" width="48%" />
  
  <img src="https://github.com/user-attachments/assets/22d8fcd2-32f4-41c5-acb6-7eedf1bc66ee" width="48%" />
</p>
<div align="center">
  <a href="#Installation"><b>Installation</b></a> |
  <a href="https://arxiv.org/abs/2506.07823"><b>PrePrint</b></a> |
  <a href="https://youtu.be/zquKLxbAU_Y"><b>Video</b></a> |
  
</div>


## Features
**MPX** is a [JAX](https://github.com/google/jax) library that provides:

✅ **True GPU Parallelism**
Exploits both temporal and state-space parallel scans directly on the GPU, without approximations or offline precomputations. Lower the complexity to $\mathcal{O}(\log^2{n}\log{N} + \log^2{m})$  from the classical $\mathcal{O}(N(n + m)^3)$ where n = state dim, m = control dim, N = horizon length

✅ **JAX Autodiff & Vectorization**
Fully differentiable solver easily integrates into learning pipelines and supports batched RL-style environments.

✅ **A multiple-shooting SQP** formulation solves the KKT system in parallel, maintaining exactness and fast convergence.

✅ **MJX MODELS** Support [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) whole body dynamics (included examples with [**Talos**](https://github.com/iit-DLSLab/mpx/blob/main/examples/mjx_talos.py), [**H1**](https://github.com/iit-DLSLab/mpx/blob/main/examples/mjx_h1.py), [**Aliengo**](https://github.com/iit-DLSLab/mpx/blob/main/examples/mjx_quad.py) and **Go2**)

The solver is wrapped by the `MPCControllerWrapper` class, and all the settings (such as the dynamics model and cost function to be used) can be changed in the config files. Examples for various legged robots are provided in the `examples` folder.
> **Note:**  
> If you want to solve multiple MPC in parallel, look at the examples/multi_env.py
> `MPCWrapper` is designed to use the whole body model, if you want to use the srbd model, use `mpc_wrapper_srbd.py`; look at examples/srbd_quad.py

## Task examples
| Acrobot Swing-Up | Quadruped Trot | Humanoid Jump | Quadruped Barrel roll |
|---|---|---|---|
| <img src="https://github.com/user-attachments/assets/af15576c-8fab-4e53-ac06-8f9e648703f6" width="100%" /> | <img src="https://github.com/user-attachments/assets/51f7eb3e-b344-4a92-9b16-837ca5dc71c6" width="100%" /> | <img src="https://github.com/user-attachments/assets/7b39eef5-a7d5-4243-a590-a6dab0b12af2" width="100%" /> | <img src="https://github.com/user-attachments/assets/7a875ce6-ea40-467a-b732-f473e5f40a02" width="100%" /> |
> **Note:**  
You can switch between two solvers, Primal-dual LQR of GPU-FDDP. Just change the flang in the config `solver_mode = "fddp" or "primal_dual`. 
## Installation

### Prerequisites
- Install [Pixi](https://pixi.prefix.dev/latest/) - a fast conda package manager

### Clone the repo
```
git clone git@github.com:iit-DLSLab/mpx.git
cd mpx && git submodule update --init --recursive
```

### Set Up Environment with Pixi
Pixi will automatically manage your environment with all dependencies including ROS 2 Humble, JAX, and build tools:

```bash
# Initialize the Pixi environment
pixi install

# Activate the environment
pixi shell
```

### Alternative: Traditional Conda Setup (Deprecated)
For reference, the legacy conda setup is:
```bash
conda create -n mpx_env python=3.13 -y
conda activate mpx_env
pip install -e .
```


## RUN example
```bash
# Using Pixi
pixi run python mpx/examples/mjx_quad.py

# Or activate the environment first
pixi shell
python mpx/examples/mjx_quad.py
```

Use the keyboard's arrows to control the robot.

## Diffusion-MPPI-guided FDDP

MPX includes a receding-horizon manipulation solver that anneals a diagonal
Gaussian in control-knot space, then refines its mean with FDDP. The local
solver adds a temporary Gaussian precision to `l_u` and `l_uu`, decays that
guidance each iteration, and reports physical task cost separately from prior
cost. Original FDDP and primal-dual modes remain available.

```bash
# Interactive Push-T with live samples, mean, FDDP prediction, and history
pixi run python -m mpx.examples.push_t_guided_mpc --viewer --verbose

# Deterministic Push-T benchmark and plots
pixi run python -m mpx.examples.push_t_guided_mpc \
  --headless --seeds 0 1 2 3 4 --plot

# Interactive full-MJX AgileX Piper box push
pixi run python -m mpx.examples.agilex_box_push_guided_mpc --viewer --verbose

# Nominal and mismatched-plant Piper benchmarks
pixi run python -m mpx.examples.agilex_box_push_guided_mpc \
  --headless --seeds 0 1 2 3 4 --condition nominal --plot
pixi run python -m mpx.examples.agilex_box_push_guided_mpc \
  --headless --condition heavy
```

Both examples accept `--mode fddp`, `--mode mppi`, `--mode mppi_fddp`, or
`--mode guided`. See [`docs/diffusion_mppi_fddp.md`](docs/diffusion_mppi_fddp.md)
for equations, ablations, video commands, diagnostics, model attribution, and
measured limitations.

> **Note:**  
The first time running the script it can take more than a minute to JIT the solver

## Citing this work

```bibtex
@article{amatuccisousa26ral,
 author={Amatucci, Lorenzo and Sousa-Pinto, João and Turrisi, Giulio and Orban, Dominique and Barasuol, Victor and Semini, Claudio},
 title={Primal-Dual iLQR for GPU-Accelerated Learning and Control in Legged Robots},
 year={2026},
 volume={11},
 number={1},
 pages={1010-1017},
 journal={IEEE Robotics and Automation Letters},
 doi={10.1109/LRA.2025.3632610}
}
```
