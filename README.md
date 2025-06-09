# MPX
Model Predictive Control in JAX

This repo implements the code for legged robot mpc and trajectory Optimization all in jax. 

 The solver is wrapped by the `MPCControllerWrapper` class, and all the settings (such as the dynamics model and cost function to be used) can be changed in the config files. Examples for various legged robot are provided in the `examples` folder.
> **Note:**  
> If you want to solve multile MPC in parallel use `BatchedMPCControllerWrapper` look at the examples/multi_env.py
> `MPCControllerWrapper` and `BatchedMPCControllerWrapper` are designed for to use the whole body model if you want to the srbd model use `mpc_wrapper_srbd.py` look at examples/srbd_quad.py

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


