<p align="left">
 <img src="https://github.com/user-attachments/assets/faaee057-131d-47da-b841-8832d536e5c5" width="70%" /> 
</p>

This repo implements the code for legged robot MPC and Trajectory Optimization all in JAX. 

> **Experimental GRiD backend warning**
>
> Branch `experimetal_GriD_support` contains experimental support for using
> [GRiD](https://github.com/robot-acceleration/GRiD) generated CUDA dynamics in
> MPX. The default backend remains MJX unless a config or example explicitly
> selects `dynamics_backend="grid"`. Treat the GRiD path as research code:
> robot-specific CUDA libraries must be generated and rebuilt for the local
> CUDA/JAX environment before running examples.
>
> Quick setup from a fresh checkout:
>
> ```bash
> git submodule update --init --recursive
> micromamba create -n mpx_env python=3.13 -y
> micromamba run -n mpx_env pip install -e .
> ```
>
> Build or rebuild the Z1 floating dynamics library from the MPX repository root:
>
> ```bash
> micromamba run -n mpx_env python -m mpx.grid_codegen.mjcf_to_urdf \
>   --mjcf mpx/data/unitree_z1/z1_floating.xml \
>   --output build/grid/z1_floating/z1_floating_grid.urdf \
>   --fixed-target ""
>
> micromamba run -n mpx_env python -m mpx.grid_codegen.generate \
>   --urdf build/grid/z1_floating/z1_floating_grid.urdf \
>   --robot-name z1_floating \
>   --out-dir build/grid/z1_floating \
>   --floating-base
>
> micromamba run -n mpx_env nvcc -std=c++17 -shared -Xcompiler=-fPIC -O3 \
>   -gencode arch=compute_86,code=sm_86 \
>   -I build/grid/z1_floating \
>   -I "$(micromamba run -n mpx_env python - <<'PY'
> import pathlib, jaxlib
> print(pathlib.Path(jaxlib.__file__).parent / 'include')
> PY
> )" \
>   -o build/grid/z1_floating/libmpx_grid_z1_floating.so \
>   mpx/grid_codegen/z1_floating_jax_bridge.cu
> ```
>
> Build the fixed-base Z1 kinematics library used by the GRiD cost path:
>
> ```bash
> micromamba run -n mpx_env python -m mpx.grid_codegen.mjcf_to_urdf \
>   --mjcf mpx/data/unitree_z1/z1.xml \
>   --output build/grid/z1_fixed/z1_grid.urdf \
>   --fixed-target end_effector
>
> micromamba run -n mpx_env python -m mpx.grid_codegen.generate \
>   --urdf build/grid/z1_fixed/z1_grid.urdf \
>   --robot-name z1 \
>   --out-dir build/grid/z1_fixed \
>   --fixed-targets end_effector
>
> micromamba run -n mpx_env nvcc -std=c++17 -shared -Xcompiler=-fPIC -O3 \
>   -gencode arch=compute_86,code=sm_86 \
>   -I build/grid/z1_fixed \
>   -I "$(micromamba run -n mpx_env python - <<'PY'
> import pathlib, jaxlib
> print(pathlib.Path(jaxlib.__file__).parent / 'include')
> PY
> )" \
>   -o build/grid/z1_fixed/libmpx_grid_z1.so \
>   mpx/grid_codegen/z1_jax_bridge.cu
> ```
>
> Adjust the `-gencode` flag for the local GPU. Generated libraries are local
> build artifacts under `build/grid/`. To run the Z1 floating example with GRiD dynamics:
>
> ```bash
> micromamba run -n mpx_env python mpx/examples/mjx_z1_floating.py \
>   --dynamics-backend grid --cost-kinematics-backend grid
> ```
>
> To compare against MJX:
>
> ```bash
> micromamba run -n mpx_env python mpx/examples/mjx_z1_floating.py \
>   --dynamics-backend mjx --cost-kinematics-backend mjx
> ```

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

### Clone the repo
```
git clone git@github.com:iit-DLSLab/mpx.git
cd mpx && git submodule update --init --recursive
```

### Set Up Conda Environment
Create and activate the conda environment:
```
conda create -n mpx_env python=3.13 -y
conda activate mpx_env
```

### Install with pip 
from the repo main folder
```
pip install -e .
```


## RUN example
```
conda activate mpx_env
python mpx/examples/mjx_quad.py
## Use the keyboard's arrows to control the robot ##
```

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
