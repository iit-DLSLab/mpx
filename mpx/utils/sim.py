from __future__ import annotations

from typing import Any, Sequence

try:
    import glfw
except ImportError:  # pragma: no cover - only needed for interactive viewers
    glfw = None

import mujoco
import numpy as np


def geom_ids(model: mujoco.MjModel, names: Sequence[str]) -> np.ndarray:
    """Return the MuJoCo geom ids for the provided geom names."""

    return np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names],
        dtype=np.int32,
    )


def geom_positions(data: mujoco.MjData, geom_ids: Sequence[int], flatten: bool = True) -> np.ndarray:
    """Return the geom positions for the selected geoms."""

    positions = np.asarray([data.geom_xpos[int(geom_id)] for geom_id in geom_ids], dtype=np.float64)
    return positions.reshape(-1) if flatten else positions


def estimate_contacts(
    data: mujoco.MjData,
    contact_geom_ids: Sequence[int],
    dist_threshold: float = 0.0,
) -> np.ndarray:
    """Estimate binary contact state for a set of contact geoms from MuJoCo contacts."""

    contact_geom_ids = np.asarray(contact_geom_ids, dtype=np.int32)
    contact_state = np.zeros(contact_geom_ids.shape[0], dtype=np.float32)
    geom_to_contact = {int(geom_id): idx for idx, geom_id in enumerate(contact_geom_ids)}

    for idx in range(data.ncon):
        contact = data.contact[idx]
        if contact.dist > dist_threshold:
            continue
        geom1 = geom_to_contact.get(int(contact.geom1))
        geom2 = geom_to_contact.get(int(contact.geom2))
        if geom1 is not None:
            contact_state[geom1] = 1.0
        if geom2 is not None:
            contact_state[geom2] = 1.0

    return contact_state


def estimate_named_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_names: Sequence[str],
    dist_threshold: float = 0.0,
) -> np.ndarray:
    """Estimate binary contact state directly from config-style geom names."""

    return estimate_contacts(data, geom_ids(model, geom_names), dist_threshold=dist_threshold)


class KeyboardVelocityCommand:
    """Arrow-key forward and yaw command for passive MuJoCo viewers.

    Usage:
    `command = KeyboardVelocityCommand(vx=0.3)`
    `viewer = mujoco.viewer.launch_passive(..., key_callback=command.key_callback)`
    `mpc_input = command.mpc_input(robot_height)`
    """

    def __init__(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        wz: float = 0.0,
        forward_step: float = 0.1,
        yaw_step: float = 0.2,
        forward_limits: tuple[float, float] = (-1.0, 1.0),
        yaw_limits: tuple[float, float] = (-1.5, 1.5),
    ):
        self.vx = float(vx)
        self.vy = float(vy)
        self.wz = float(wz)
        self.forward_step = float(forward_step)
        self.yaw_step = float(yaw_step)
        self.forward_limits = tuple(float(value) for value in forward_limits)
        self.yaw_limits = tuple(float(value) for value in yaw_limits)
        self._overlay_dirty = True

    def _clip(self):
        self.vx = float(np.clip(self.vx, *self.forward_limits))
        self.wz = float(np.clip(self.wz, *self.yaw_limits))

    def reset(self):
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self._overlay_dirty = True

    def key_callback(self, key: int):
        """Update the planar command from GLFW arrow keys."""

        if glfw is None:
            raise RuntimeError("glfw is required to use KeyboardVelocityCommand.")

        # Left/right arrows steer the commanded yaw rate around the vertical axis.
        if key == glfw.KEY_UP:
            self.vx += self.forward_step
        elif key == glfw.KEY_DOWN:
            self.vx -= self.forward_step
        elif key == glfw.KEY_LEFT:
            self.wz += self.yaw_step
        elif key == glfw.KEY_RIGHT:
            self.wz -= self.yaw_step
        elif key in (glfw.KEY_SPACE, glfw.KEY_ENTER, glfw.KEY_BACKSPACE):
            self.reset()
        else:
            return

        self._clip()
        self._overlay_dirty = True

    def planar_command(self) -> np.ndarray:
        """Return the current planar command `[vx, vy]`."""

        return np.array([self.vx, self.vy], dtype=np.float64)

    def mpc_input(self, robot_height: float) -> np.ndarray:
        """Return the 7D locomotion command used by the MPC examples."""

        return np.array(
            [self.vx, self.vy, 0.0, 0.0, 0.0, self.wz, robot_height],
            dtype=np.float64,
        )

    def overlay_text(self) -> tuple[str, str]:
        """Return short viewer text showing controls and the current command."""

        return (
            "Up/Down: forward | Left/Right: yaw | Space: stop",
            f"vx {self.vx:+.2f}  wz {self.wz:+.2f}",
        )

    def consume_overlay_text(self) -> tuple[str, str] | None:
        """Return overlay text only when the command was updated."""

        if not self._overlay_dirty:
            return None
        self._overlay_dirty = False
        return self.overlay_text()


def _reserve_user_geom(viewer) -> int:
    if viewer is None:
        return -1
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        raise ValueError(
            f"Viewer user scene is full ({viewer.user_scn.ngeom}/{viewer.user_scn.maxgeom})."
        )
    viewer.user_scn.ngeom += 1
    return viewer.user_scn.ngeom - 1


def render_vector(
    viewer,
    vector: np.ndarray,
    pos: np.ndarray,
    scale: float,
    color: np.ndarray = np.array([1.0, 0.0, 0.0, 1.0]),
    geom_id: int = -1,
) -> int:
    """Render a decorative arrow aligned with the provided vector."""

    if viewer is None:
        return -1

    if geom_id < 0:
        geom_id = _reserve_user_geom(viewer)

    geom = viewer.user_scn.geoms[geom_id]
    direction = np.asarray(vector, dtype=np.float64).reshape(3)
    start = np.asarray(pos, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([0.0, 0.0, 1.0])
        norm = 1.0

    end = start + (scale * direction / norm)
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.ones(3) * 1e-3,
        pos=np.zeros(3),
        mat=np.eye(3).reshape(9),
        rgba=np.asarray(color, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.01, start, end)
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    geom.objid = -1
    return geom_id


def render_sphere(
    viewer,
    position: np.ndarray,
    diameter: float,
    color: np.ndarray = np.array([1.0, 0.0, 0.0, 1.0]),
    geom_id: int = -1,
) -> int:
    """Render a decorative sphere at the provided position."""

    if viewer is None:
        return -1

    if geom_id < 0:
        geom_id = _reserve_user_geom(viewer)

    geom = viewer.user_scn.geoms[geom_id]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([0.5 * diameter, 0.0, 0.0]),
        pos=np.asarray(position, dtype=np.float64).reshape(3),
        mat=np.eye(3).reshape(9),
        rgba=np.asarray(color, dtype=np.float32),
    )
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    geom.objid = -1
    return geom_id


def render_sphere_trajectory(
    viewer,
    positions: np.ndarray,
    alphas: np.ndarray,
    diameter: float,
    color: np.ndarray = np.array([1.0, 0.45, 0.0, 1.0]),
    geom_ids: list[int] | None = None,
) -> list[int]:
    """Render or update a sequence of decorative spheres."""

    if viewer is None:
        return []

    positions = np.asarray(positions, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64)
    if positions.shape[0] == 0:
        return []

    if geom_ids is None or len(geom_ids) != positions.shape[0]:
        geom_ids = [-1] * positions.shape[0]

    base_color = np.asarray(color, dtype=np.float32)
    for idx, (position, alpha) in enumerate(zip(positions, alphas, strict=False)):
        rgba = np.array(base_color, copy=True)
        rgba[3] = float(alpha)
        geom_ids[idx] = render_sphere(
            viewer,
            position,
            diameter,
            color=rgba,
            geom_id=geom_ids[idx],
        )

    return geom_ids


def _build_ghost_geoms(
    viewer,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
) -> dict[int, dict[str, Any]]:
    """Cache the static visual spec for each rendered ghost geom."""

    scene = mujoco.MjvScene(mj_model, maxgeom=max(2 * mj_model.ngeom, 200))
    mujoco.mjv_updateScene(
        mj_model,
        mj_data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )

    ghost_geoms = {}
    ignored_names = {"floor", "plane", "world", "ground"}
    for geom in scene.geoms[: scene.ngeom]:
        if geom.segid == -1:
            continue

        geom_model_id = int(geom.objid)
        geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_model_id)
        body_id = mj_model.geom_bodyid[geom_model_id]
        body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        geom_rgba = mj_model.geom_rgba[geom_model_id]
        if geom_name in ignored_names or body_name in ignored_names or geom_rgba[3] == 0:
            continue

        ghost_geoms[_reserve_user_geom(viewer)] = {
            "model_id": geom_model_id,
            "type": int(geom.type),
            "size": np.array(geom.size, copy=True),
            "rgba": np.array(geom.rgba, copy=True),
            "dataid": int(geom.dataid),
            "emission": float(geom.emission),
            "specular": float(geom.specular),
            "shininess": float(geom.shininess),
        }

    return ghost_geoms


def render_ghost_robot(
    viewer,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    alpha: float = 0.5,
    ghost_geoms: dict[int, dict[str, Any]] | None = None,
) -> dict[int, dict[str, Any]]:
    """Render or update a translucent ghost robot in a passive MuJoCo viewer."""

    if viewer is None:
        return {}

    if ghost_geoms is None or len(ghost_geoms) == 0:
        # Build the cache once from the model geoms, then only update transforms.
        ghost_geoms = _build_ghost_geoms(viewer, mj_model, mj_data)

    for scn_id, geom in ghost_geoms.items():
        geom_model_id = geom["model_id"]
        rgba = np.array(geom["rgba"], copy=True)
        rgba[3] = alpha

        decorative_geom = viewer.user_scn.geoms[scn_id]
        mujoco.mjv_initGeom(
            decorative_geom,
            type=geom["type"],
            rgba=rgba,
            size=geom["size"],
            pos=mj_data.geom_xpos[geom_model_id],
            mat=mj_data.geom_xmat[geom_model_id].reshape(9),
        )
        decorative_geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        decorative_geom.segid = -1
        decorative_geom.objid = -1
        decorative_geom.dataid = geom["dataid"]
        decorative_geom.emission = geom["emission"]
        decorative_geom.specular = geom["specular"]
        decorative_geom.shininess = geom["shininess"]
        decorative_geom.reflectance = 0.0

    return ghost_geoms


def render_ghost_trajectory(
    viewer,
    mj_model: mujoco.MjModel,
    qpos_sequence: np.ndarray,
    alphas: np.ndarray,
    ghost_geoms: list[dict[int, dict[str, Any]] | None] | None = None,
    scratch_data: mujoco.MjData | None = None,
    subsample: int = 20,
) -> tuple[list[dict[int, dict[str, Any]]], mujoco.MjData]:
    """Render a sequence of ghost robots, typically used for planned trajectories."""

    qpos_sequence = np.asarray(qpos_sequence)
    alphas = np.asarray(alphas)
    if subsample > 1:
        qpos_sequence = qpos_sequence[::subsample]
        alphas = alphas[::subsample]

    if scratch_data is None:
        scratch_data = mujoco.MjData(mj_model)
    if ghost_geoms is None or len(ghost_geoms) != len(qpos_sequence):
        ghost_geoms = [None] * len(qpos_sequence)

    for idx, (qpos, alpha) in enumerate(zip(qpos_sequence, alphas, strict=False)):
        scratch_data.qpos = qpos
        mujoco.mj_forward(mj_model, scratch_data)
        ghost_geoms[idx] = render_ghost_robot(
            viewer,
            mj_model,
            scratch_data,
            alpha=float(alpha),
            ghost_geoms=ghost_geoms[idx],
        )

    return ghost_geoms, scratch_data


class VideoRecorder:
    """Offscreen mp4 recorder built around `mujoco.Renderer`.

    Designed for headless execution of the online MPC examples (mjx_quad,
    mjx_h1, ...). Tracks the robot base by default (lookat = qpos[:3]).

    Requires:
      * `MUJOCO_GL=egl` (or another working backend) set BEFORE `import mujoco`
        — see `enable_offscreen_gl_for_video()`.
      * `imageio[ffmpeg]` for libx264 mp4 output.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        path: str,
        *,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        distance: float = 3.0,
        azimuth: float = 90.0,
        elevation: float = -20.0,
        bit_depth: int = 10,
    ):
        import imageio  # late import: only needed when recording

        self._renderer = mujoco.Renderer(model, height=height, width=width)
        
        # Pick pixel format based on bit depth
        pix_fmt = "yuv420p10le" if bit_depth == 10 else "yuv420p"
        
        self._writer = imageio.get_writer(
            path,
            format="FFMPEG",
            codec="libx264",
            fps=fps,
            macro_block_size=1,
            output_params=["-pix_fmt", pix_fmt],
        )
        self._cam = mujoco.MjvCamera()
        self._cam.distance = float(distance)
        self._cam.azimuth = float(azimuth)
        self._cam.elevation = float(elevation)
        self._cam.lookat[:] = [0.0, 0.0, 0.0]

    def capture(self, data: mujoco.MjData, lookat: np.ndarray | None = None) -> None:
        """Render and append one frame.

        Default `lookat` is the floating-base world position (`qpos[:3]`) when
        the model has at least 3 generalised coords — the typical case for
        legged-robot configs. For low-dimensional models (e.g. acrobot's
        2-DOF qpos), falls back to the world body's xpos so the camera stays
        pointed at something sensible instead of crashing on the reshape.
        """

        if lookat is None:
            qpos = np.asarray(data.qpos, dtype=np.float64)
            if qpos.size >= 3:
                lookat = qpos[:3]
            else:
                # Centre on the first non-world body's world position. Always 3D.
                lookat = np.asarray(data.xpos[1] if data.xpos.shape[0] > 1 else data.xpos[0],
                                    dtype=np.float64)
        self._cam.lookat[:] = np.asarray(lookat, dtype=np.float64).reshape(3)
        self._renderer.update_scene(data, self._cam)
        self._writer.append_data(self._renderer.render())

    def close(self) -> None:
        try:
            self._writer.close()
        finally:
            self._renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
