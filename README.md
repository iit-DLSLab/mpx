# mpx
Model Predictive Control in JAX
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
$ pip install git+https://github.com/google/trajax
```
[Trajax GitHub](https://github.com/username/trajax)

### Quadruped Simulation Setup
To run the simulation with the quadruped, install gymquadruped:

[DLS-iit gym-quadruped](https://github.com/iit-DLSLab/gym-quadruped)
