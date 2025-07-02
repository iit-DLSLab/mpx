from setuptools import setup, find_packages

setup(
    name='mpx',
    version='0.1',
    description='Model Predictive Control in JAX',
    url='https://github.com/iit-DLSLab/mpx',
    author='Amatucci Lorenzo',
    license='BSD 3-clause',
     packages=['mpx'],
    install_requires=['jax[cuda12]==0.6.1',
                      'mujoco==3.3.0',
                      'mujoco-mjx==3.3.0',
                      'gym-quadruped==0.0.6']
)