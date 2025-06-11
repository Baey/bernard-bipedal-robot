import bpy
import numpy as np
import mathutils
import math
import csv
from mathutils import Matrix
from scipy.spatial.transform import Rotation as R


# DOFs and bodies mappings and settings
dof_map = {
    "r_hip_joint": "z",
    "l_hip_joint": "z",
    "r_arm_joint": "x",
    "l_arm_joint": "x",
    "r_knee_joint": "x",
    "l_knee_joint": "x",
}
dof_idx = {
    "r_hip_joint": 0,
    "l_hip_joint": 1,
    "r_arm_joint": 2,
    "l_arm_joint": 3,
    "r_knee_joint": 4,
    "l_knee_joint": 5,
}
axis_index = {"x": 0, "y": 1, "z": 2}
body_bone_names = [
    "body", "r_hip", "l_hip",
    "r_arm", "l_arm", 
    "r_forearm", "l_forearm",
    "r_foot", "l_foot"
]
body_idx = {
    "body": 0,
    "r_hip": 1,
    "l_hip": 2,
    "r_arm": 3,
    "l_arm": 4, 
    "r_forearm": 5,
    "l_forearm": 6,
    "r_foot": 7,
    "l_foot": 8
}
dof_to_body = {
    "body": "body",
    "r_hip_joint": "r_hip",
    "l_hip_joint": "l_hip",
    "r_arm_joint": "r_arm",
    "l_arm_joint": "l_arm",
    "r_knee_joint": "r_forearm",
    "l_knee_joint": "l_forearm",
    "r_foot_joint": "r_foot",
    "l_foot_joint": "l_foot"
}
dof_signs = {
    name: -1.0 if name in {"l_arm_joint", "r_knee_joint"} else 1.0 for name in dof_map
}
dof_offset = {
    "r_hip_joint": 0.0,
    "l_hip_joint": 0.0,
    "r_arm_joint": 1.57079633,
    "l_arm_joint": -1.57079633,
    "r_knee_joint": -1.57079633,
    "l_knee_joint": 1.57079633,
}
fix_body_rot = {
    "body": mathutils.Euler((math.radians(270), 0, 0), 'XYZ').to_quaternion(),
    "r_hip_joint": mathutils.Euler((math.radians(270), 0, 0), 'XYZ').to_quaternion(),
    "l_hip_joint": mathutils.Euler((math.radians(270), 0, 0), 'XYZ').to_quaternion(),
    "r_arm_joint": mathutils.Euler((math.radians(270), 0, 0), 'XYZ').to_quaternion(),
    "l_arm_joint": mathutils.Euler((math.radians(270), 0, 0), 'XYZ').to_quaternion(),
    "r_knee_joint": mathutils.Euler((math.radians(270), 0, 0), 'XYZ').to_quaternion(),
    "l_knee_joint": mathutils.Euler((math.radians(270), 0, 0), 'XYZ').to_quaternion(),
    "r_foot_joint": mathutils.Euler((math.radians(270), 0, 0), 'XYZ').to_quaternion(),
    "l_foot_joint": mathutils.Euler((math.radians(270), 0, 0), 'XYZ').to_quaternion(),
}


# Retrieve pose bones
armature_name = "Armature"
arm = bpy.data.objects[armature_name]
pose_bones = arm.pose.bones
action_name = arm.animation_data.action.name

text_block = bpy.data.texts.get("Log") or bpy.data.texts.new("Log")
text_block.write(f"Exporting {action_name} animation data...\n")

# Retrieve scene settings
fps = bpy.context.scene.render.fps
start_frame = bpy.context.scene.frame_start
end_frame = bpy.context.scene.frame_end
frames = list(range(start_frame, end_frame + 1))
N = len(frames)
D = len(dof_map)
B = len(body_bone_names)
text_block.write(f"FPS: {fps}\n")
text_block.write(f"Frames: {N}\n")
text_block.write(f"DOFs: {D}\n")
text_block.write(f"Bodies: {B}\n")

# Prepare buffers
dof_names = np.array(list(dof_map.keys()), dtype='<U32')
body_names = np.array(body_bone_names, dtype='<U32')

dof_positions = np.zeros((N, D), dtype=np.float32)
dof_velocities = np.zeros((N, D), dtype=np.float32)

body_positions = np.zeros((N, B, 3), dtype=np.float32)
body_rotations = np.zeros((N, B, 4), dtype=np.float32)  # wxyz
body_linear_velocities = np.zeros((N, B, 3), dtype=np.float32)
body_angular_velocities = np.zeros((N, B, 3), dtype=np.float32)

# Prepare CSV data
dof_positions_csv = []
header = []
for bone in pose_bones:
    name = bone.name
    if name in dof_map:
        header.append(f"{name}.{dof_map[name]}")

text_block.write("Iterating through frames...\n")
bpy.ops.object.mode_set(mode='POSE')
for i, frame in enumerate(frames):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    frame_data = []

    # Retrieve pose bones for the current frame
    for bone in pose_bones:
        print(bone)
        if bone is None: continue
        if bone.name in dof_map:
            axis = dof_map[bone.name]
            j = dof_idx[bone.name]
            rot = bone.matrix.to_euler()
            dof_positions[i, j] = rot[axis_index[axis]] * dof_signs[bone.name] - dof_offset[bone.name]
            if bone.name == "l_knee_joint":
                arm_idx = dof_idx["l_arm_joint"]
                dof_positions[i, j] += dof_positions[i, arm_idx]
            if bone.name == "r_knee_joint":
                arm_idx = dof_idx["r_arm_joint"]
                dof_positions[i, j] += dof_positions[i, arm_idx]

            frame_data.append(dof_positions[i, j])
        if bone.name in dof_to_body:
            body_name = dof_to_body[bone.name]
            j = body_idx[body_name]
            mat = arm.matrix_world @ bone.matrix
            pos = arm.matrix_world @ bone.head
#            quat = (mat.to_quaternion() @ fix_body_rot[bone.name]).normalized()
            quat = (bone.matrix.to_quaternion() @ fix_body_rot[bone.name]).normalized()
#            parent_matrix = bone.parent.matrix if bone.parent else Matrix.Identity(4)
#            local_matrix = parent_matrix.inverted() @ bone.matrix
#            quat = (local_matrix.to_quaternion() @ fix_body_rot[bone.name]).normalized()
            body_positions[i, j] = pos[:]
            body_rotations[i, j] = np.array([quat.x, quat.y, quat.z, quat.w])
    dof_positions_csv.append(frame_data)

text_block.write("All frames processed!\n")
text_block.write("Calculating linear velocities...\n")
dt = 1.0 / fps
dof_velocities[1:-1] = (dof_positions[2:] - dof_positions[:-2]) / (2 * dt)
dof_velocities[0] = dof_velocities[1]
dof_velocities[-1] = dof_velocities[-2]

body_linear_velocities[1:-1] = (body_positions[2:] - body_positions[:-2]) / (2 * dt)
body_linear_velocities[0] = body_linear_velocities[1]
body_linear_velocities[-1] = body_linear_velocities[-2]

text_block.write("Calculating angular velocities...\n")
for i in range(1, N - 1):
    for j in range(B):
        q_prev = body_rotations[i - 1, j]
        q_next = body_rotations[i + 1, j]

        q1 = R.from_quat(q_prev)
        q2 = R.from_quat(q_next)

        q_delta = q2 * q1.inv()
        ang_vel = q_delta.as_rotvec() / (2 * dt)
        body_angular_velocities[i, j] = ang_vel

body_angular_velocities[0] = body_angular_velocities[1]
body_angular_velocities[-1] = body_angular_velocities[-2]

text_block.write("All velocities calculated!\n")
text_block.write("Exporting data to .npz file...\n")
np.savez(f"/Users/blazejszargut/Documents/bernard-bipedal-robot/motion/dataset/bernard_{action_name}.npz",
         fps=np.int64(fps),
         dof_names=dof_names,
         body_names=body_names,
         dof_positions=dof_positions,
         dof_velocities=dof_velocities,
         body_positions=body_positions,
         body_rotations=body_rotations,
         body_linear_velocities=body_linear_velocities,
         body_angular_velocities=body_angular_velocities)

with open(f"/Users/blazejszargut/Documents/bernard-bipedal-robot/motion/dataset/bernard_{action_name}_dof_positions.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(dof_positions_csv)
