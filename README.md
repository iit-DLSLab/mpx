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
Exploits both temporal and state-space parallel scans directly on the GPU, without approximations or offline precomputations. Lower the complexity to $\mathcal{O}(n\log{N} + m)$  from the classical $\mathcal{O}(N(n + m)^3)$ where n = state dim, m = control dim, N = horizon length

✅ **JAX Autodiff & Vectorization**
Fully differentiable solver easily integrates into learning pipelines and supports batched RL-style environments.

✅ **A multiple-shooting SQP** formulation solves the KKT system in parallel, maintaining exactness and fast convergence.

✅ **MJX MODELS** Support [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) whole body dynamics (included examples with [**Talos**](https://github.com/iit-DLSLab/mpx/blob/main/examples/mjx_talos.py), [**H1**](https://github.com/iit-DLSLab/mpx/blob/main/examples/mjx_h1.py), [**Aliengo**](https://github.com/iit-DLSLab/mpx/blob/main/examples/mjx_quad.py) and **Go2**)

The solver is wrapped by the `MPCControllerWrapper` class, and all the settings (such as the dynamics model and cost function to be used) can be changed in the config files. Examples for various legged robots are provided in the `examples` folder.
> **Note:**  
> If you want to solve multiple MPC in parallel, use `BatchedMPCControllerWrapper` look at the examples/multi_env.py
> `MPCControllerWrapper` and `BatchedMPCControllerWrapper` are designed to use the whole body model, if you want to use the srbd model, use `mpc_wrapper_srbd.py`; look at examples/srbd_quad.py

## Installation

### Set Up Conda Environment
Create and activate the conda environment:
```
conda create -n mpx_env python=3.13 -y
conda activate mpx_env
```

### Install CUDA-Enabled JAX
Install the CUDA version of JAX:
```
pip install --upgrade pip
pip install -U "jax[cuda12]"
```

### Install Mujoco and Trajax
Install Mujoco:
```
pip install mujoco
pip install mujoco-mjx
```

Install Trajax (see the online repository for more details):
```
pip install git+https://github.com/google/trajax
```
[Trajax GitHub](https://github.com/username/trajax)

### Quadruped Simulation Setup
To run the simulation with the quadruped, install gymquadruped:

[DLS-iit gym-quadruped](https://github.com/iit-DLSLab/gym-quadruped)

## RUN example
```
conda activate mpx_env
python mpx/examples/mjx_quad.py
```
> **Note:**  
The first time running the script it can take more than a minute to JIT the solver

## Citing this work

```bibtex
@misc{2025primaldualilqrgpuacceleratedlearning,
      title={Primal-Dual iLQR for GPU-Accelerated Learning and Control in Legged Robots}, 
      author={Lorenzo Amatucci and João Sousa-Pinto and Giulio Turrisi and Dominique Orban and Victor Barasuol and Claudio Semini},
      year={2025},
      eprint={2506.07823},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.07823}, 
}
```
