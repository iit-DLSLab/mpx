import mujoco 
from mujoco import mjx 
import jax 
# import primal_dual_ilqr.utils.mpc_utils as mpc_utils
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
from timeit import default_timer as timer



model = mujoco.MjModel.from_xml_path('../mujoco_menagerie/unitree_go2/scene_mjx.xml')
data = mujoco.MjData(model)
mjx_model = mjx.put_model(model)


jit_step = jax.jit(mjx.step)
frames = []
mujoco.mj_resetData(model, data)
mjx_data = mjx.put_data(model, data)
qpos = jax.numpy.array([0, 0, 0.27, 1, 0, 0, 0, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
rng = jax.numpy.tile(qpos, (4096, 1))
batch = jax.vmap(lambda rng: mjx_data.replace(qpos=rng))(rng)

jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
batch = jit_step(mjx_model, batch)
start = timer()
batch = jit_step(mjx_model, batch)
end = timer()
print(f"Time elapsed: {end - start} s")