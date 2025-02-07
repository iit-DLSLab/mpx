import jax.numpy as jnp
import jax

def quaternion_product(q1, q2):
    w1 = q1[0]
    x1 = q1[1]
    y1 = q1[2]
    z1 = q1[3]
    w2 = q2[0]
    x2 = q2[1]
    y2 = q2[2]
    z2 = q2[3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return jnp.array([w,x, y, z])
def quaternion_integration(w, q, dt):
    # v = w[:3]*dt*0.5*jnp.sin(jnp.linalg.norm(w[:3])*dt/2)/(jnp.linalg.norm(w[:3])*dt/2)
    # q_dot = jnp.array([v[0], v[1], v[2], jnp.cos(jnp.linalg.norm(w[:3])*dt/2)])
    # q_next = quaternion_product(q_dot,q)
    # return jnp.where(jnp.linalg.norm(w[:3]) < 1e-4,q,q_next)
    temp = jnp.sin(jnp.linalg.norm(w[1:])*dt*0.5)*w[1:]*dt*0.5/jnp.linalg.norm(w[1:])
    qw = jnp.array([jnp.cos(jnp.linalg.norm(w)*dt/2),temp[0],temp[1],temp[2]])
    return quaternion_product(q,qw)
@jax.jit
def rpy_intgegration(w, rpy, dt):
    roll, pitch, yaw = rpy
    conj_euler_rates = jnp.array([
        [jnp.float32(1), jnp.float32(0), -jnp.sin(pitch)],
        [jnp.float32(0), jnp.cos(roll), jnp.cos(pitch) * jnp.sin(roll)],
        [jnp.float32(0), -jnp.sin(roll), jnp.cos(pitch) * jnp.cos(roll)]
    ])
    inv_conj_euler_rates = jnp.linalg.inv(conj_euler_rates)
    return rpy + jnp.dot(inv_conj_euler_rates, w) * dt
@jax.jit
def quaternion_to_rpy(q):
    """
    Converts a quaternion to roll, pitch, and yaw (RPY) angles.

    Args:
        q (jax.numpy.ndarray): A quaternion as a 4D vector [w, x, y, z].

    Returns:
        rpy (jax.numpy.ndarray): Roll, pitch, and yaw as a 3D vector [roll, pitch, yaw].
    """
    # Extract quaternion components
    w, x, y, z = q

    # Compute roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    # Compute pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = jnp.where(jnp.abs(sinp) >= 1,
                       jnp.sign(sinp) * jnp.pi / 2,  # Clamp to 90 degrees
                       jnp.arcsin(sinp))

    # Compute yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    return jnp.array([roll, pitch, yaw])

def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion using JAX.

    Args:
        R: A 3x3 rotation matrix (jax.numpy array).

    Returns:
        A quaternion as a 4-element jax.numpy array [q_w, q_x, q_y, q_z].
    """
    trace = jnp.trace(R)

    def case_trace_positive():
        S = jnp.sqrt(trace + 1.0) * 2
        q_w = 0.25 * S
        q_x = (R[2, 1] - R[1, 2]) / S
        q_y = (R[0, 2] - R[2, 0]) / S
        q_z = (R[1, 0] - R[0, 1]) / S
        return jnp.array([q_w, q_x, q_y, q_z])

    def case_r11_largest():
        S = jnp.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q_w = (R[2, 1] - R[1, 2]) / S
        q_x = 0.25 * S
        q_y = (R[0, 1] + R[1, 0]) / S
        q_z = (R[0, 2] + R[2, 0]) / S
        return jnp.array([q_w, q_x, q_y, q_z])

    def case_r22_largest():
        S = jnp.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q_w = (R[0, 2] - R[2, 0]) / S
        q_x = (R[0, 1] + R[1, 0]) / S
        q_y = 0.25 * S
        q_z = (R[1, 2] + R[2, 1]) / S
        return jnp.array([q_w, q_x, q_y, q_z])

    def case_r33_largest():
        S = jnp.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q_w = (R[1, 0] - R[0, 1]) / S
        q_x = (R[0, 2] + R[2, 0]) / S
        q_y = (R[1, 2] + R[2, 1]) / S
        q_z = 0.25 * S
        return jnp.array([q_w, q_x, q_y, q_z])

    # Determine which case to use
    return jax.lax.cond(
        trace > 0,
        case_trace_positive,
        lambda: jax.lax.cond(
            (R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2]),
            case_r11_largest,
            lambda: jax.lax.cond(
                R[1, 1] > R[2, 2],
                case_r22_largest,
                case_r33_largest
            )
        )
    )