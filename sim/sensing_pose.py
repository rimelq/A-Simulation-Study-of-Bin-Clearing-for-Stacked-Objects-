"""Constants for sensing pose, bin positions, camera parameters, and object scale.

SOURCE_BIN_POS / DEST_BIN_POS / SENSING_EEF_POS are in robot-local frame
(x forward from base, y left from base, z=0 at table surface). Robot base sits
at world x = -0.66 from TableArena + UR5e offset.
"""
import numpy as np

# UR5e joint angles for the overhead sensing pose
# shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
# IK target: EEF at world (-0.01, 0.22, 1.40), gripper straight down ([0,0,-1])
# Fingertip at z=1.23 clears container rim at z=1.06. Pos/ori error verified = 0.
SENSING_JOINTS = np.array([0.1298, -1.1257, 1.0249, -1.4701, -1.5708, 0.0004])

SENSING_EEF_POS  = np.array([0.65, 0.22, 0.60])   # 0.60 m above table (world z=1.40)
SENSING_EEF_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # gripper straight down

SOURCE_BIN_POS = np.array([0.65,  0.22, 0.0])
DEST_BIN_POS   = np.array([0.65, -0.22, 0.0])

# Half-extents must cover scaled item spread (+-0.119 in x,y)
BIN_HALF_SIZE = np.array([0.13, 0.13, 0.10])

OBJECT_SCALE = 0.25

CAMERA_FOV_DEG = 60.0
CAMERA_W = 640
CAMERA_H = 480
CAMERA_NEAR = 0.05
CAMERA_FAR = 2.0
