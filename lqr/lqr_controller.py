import os
import mujoco
import mujoco.viewer
import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt

os.chdir("model/BERNARD/urdf/")
with open("BERNARD.mjcf", "r") as f:
    xml = f.read()
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
data.qacc = 0  # Assert that there is no the acceleration.
mujoco.mj_inverse(model, data)
print(data.qfrc_inverse)

height_offsets = np.linspace(-0.002, 0.002, 4001)
vertical_forces = []
for offset in height_offsets:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.qacc = 0
    # Offset the height by `offset`.
    data.qpos[2] += offset
    mujoco.mj_inverse(model, data)
    vertical_forces.append(data.qfrc_inverse[2])

# Find the height-offset at which the vertical force is smallest.
idx = np.argmin(np.abs(vertical_forces))
best_offset = height_offsets[idx]

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
data.qacc = 0
data.qpos[2] += best_offset
qpos0 = data.qpos.copy()  # Save the position setpoint.
mujoco.mj_inverse(model, data)
qfrc0 = data.qfrc_inverse.copy()
print("desired forces:", qfrc0)

actuator_moment = np.zeros((model.nu, model.nv))
mujoco.mju_sparse2dense(
    actuator_moment,
    data.actuator_moment.reshape(-1),
    data.moment_rownnz,
    data.moment_rowadr,
    data.moment_colind.reshape(-1),
)
ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(actuator_moment)
ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.
print("control setpoint:", ctrl0)

data.ctrl = ctrl0
mujoco.mj_forward(model, data)
print("actuator forces:", data.qfrc_actuator)

mujoco.mj_resetData(model, data)
data.qpos = qpos0
data.ctrl = ctrl0

nu = model.nu  # Alias for the number of actuators.
R = np.eye(nu) * 0.006
nv = model.nv

# Get the Jacobians for the left and right foot CoMs.
jac_foot_left = np.zeros((3, nv))
mujoco.mj_jacBodyCom(model, data, jac_foot_left, None, model.body("l_foot").id)

jac_foot_right = np.zeros((3, nv))
mujoco.mj_jacBodyCom(model, data, jac_foot_right, None, model.body("r_foot").id)

# Average the two Jacobians to approximate the Jacobian at the midpoint.
jac_foot_mid = 0.5 * (jac_foot_left + jac_foot_right)

# Get the Jacobian for the root body (torso) CoM.
jac_com = np.zeros((3, nv))
mujoco.mj_jacSubtreeCom(model, data, jac_com, model.body("body").id)

# Use the difference for your balance metric.
jac_diff = jac_com - jac_foot_mid
Qbalance = jac_diff.T @ jac_diff

# Get all joint names.
joint_names = [
    model.joint(i).name
    for i in range(model.njnt)
    if model.joint(i).name not in ["l_foot_joint", "r_foot_joint"]
]

# Get indices into relevant sets of joints.
root_dofs = range(6)
body_dofs = range(6, nv)
left_leg_dofs = [
    model.joint(name).dofadr[0]
    for name in joint_names
    if "l" in name and ("hip" in name or "knee" in name or "arm" in name)
]
right_leg_dofs = [
    model.joint(name).dofadr[0]
    for name in joint_names
    if "r" in name and ("hip" in name or "knee" in name or "arm" in name)
]
balance_dofs = left_leg_dofs + right_leg_dofs
other_dofs = np.setdiff1d(body_dofs, balance_dofs)

# Cost coefficients.
BALANCE_COST = 0.1  # Balancing.
BALANCE_JOINT_COST = 2000  # Joints required for balancing.
OTHER_JOINT_COST = 0.3  # Other joints.

# Construct the Qjoint matrix.
Qjoint = np.eye(nv)
Qjoint[root_dofs, root_dofs] *= 0  # Don't penalize free joint directly.
Qjoint[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
Qjoint[other_dofs, other_dofs] *= OTHER_JOINT_COST

# Construct the Q matrix for position DoFs.
Qpos = BALANCE_COST * Qbalance + Qjoint

# No explicit penalty for velocities.
Q = np.block([[Qpos, np.zeros((nv, nv))], [np.zeros((nv, 2 * nv))]])

# Set the initial state and control.
mujoco.mj_resetData(model, data)
data.ctrl = ctrl0
data.qpos = qpos0

# Allocate the A and B matrices, compute them.
A = np.zeros((2 * nv, 2 * nv))
B = np.zeros((2 * nv, nu))
epsilon = 1e-6
flg_centered = True
mujoco.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)

# Solve discrete Riccati equation.
P = scipy.linalg.solve_discrete_are(A, B, Q, R)

# Compute the feedback gain matrix K.
K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

mujoco.mj_resetData(model, data)
data.qpos = qpos0

# Allocate position difference dq.
dq = np.zeros(model.nv)

DURATION = 20         # seconds
TOTAL_ROTATION = 15  # degrees
CTRL_RATE = 0.01  # seconds
BALANCE_STD = 0.1  # actuator units
OTHER_STD = 0.1  # actuator units

# Make new camera, set distance.
camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.distance = 2.3

# Enable contact force visualisation.
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

# Set the scale of visualized contact forces to 1cm/N.
model.vis.map.force = 0.01


# Define smooth orbiting function.
def unit_smooth(normalised_time: float) -> float:
    return 1 - np.cos(normalised_time * 2 * np.pi)


def azimuth(time: float) -> float:
    return 100 + unit_smooth(data.time / DURATION) * TOTAL_ROTATION


# Precompute some noise.
np.random.seed(1)
nsteps = int(np.ceil(DURATION / model.opt.timestep))
perturb = np.random.randn(nsteps, nu)

# Scaling vector with different STD for "balance" and "other"
CTRL_STD = np.empty(nu)
for i in range(nu):
    joint = model.actuator(i).trnid[0]
    dof = model.joint(joint).dofadr[0]
    CTRL_STD[i] = BALANCE_STD if dof in balance_dofs else OTHER_STD


# Smooth the noise.
width = int(nsteps * CTRL_RATE / DURATION)
kernel = np.exp(-0.5 * np.linspace(-3, 3, width) ** 2)
kernel /= np.linalg.norm(kernel)
for i in range(nu):
    perturb[:, i] = np.convolve(perturb[:, i], kernel, mode="same")

# Petla symulacyjna
step = 0
with mujoco.viewer.launch_passive(model, data, show_left_ui=False) as viewer:
    start = time.time()
    viewer._user_scn = scene_option
    while viewer.is_running():
        current = time.time()

        # Step the simulation.
        if current - start > 0.01:  # 100 Hz
            start = current
            mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
            dx = np.hstack((dq, data.qvel)).T

            # LQR control law.
            data.ctrl = np.clip(ctrl0 - K @ dx, -9, 9)
            # data.ctrl = ctrl0 - K @ dx
            data.ctrl += CTRL_STD * perturb[step]
            step += 1

            # Step the simulation.
            mujoco.mj_step(model, data)

            mujoco.mj_step(model, data)
            viewer.sync()
