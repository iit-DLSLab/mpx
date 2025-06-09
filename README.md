# MPX - Model Predictive Control in JAX
![Image](https://github.com/user-attachments/assets/de8b9650-684e-4f31-82e4-9a0035f50f8e)
![Image](https://github.com/user-attachments/assets/b7d7ab13-9e3b-4cc1-acf3-200ae3697af8)
<div align="center">
  <a href="#Installation"><b>Installation</b></a> |
  <a href=""><b>PrePrint</b></a> |
  <a href="https://www.youtube.com/watch?v=Yar4W-Vlh2A"><b>Video</b></a> |
  
</div>

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

## Citing this work

```bibtex
@INPROCEEDINGS{mpc,
      title={Primal-Dual iLQR for GPU-Accelerated Learning and Control in Legged Robots}, 
      author={Lorenzo Amatucci and João Sousa-Pinto and Giulio Turrisi and, Dominique Orban and  Victor Barasuol and Claudio Semini},
       booktitle={},
  year={2025}
}
```
