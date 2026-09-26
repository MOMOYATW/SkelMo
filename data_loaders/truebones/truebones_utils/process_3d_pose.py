import BVH
from Animation import *
from InverseKinematics import animation_from_positions
import numpy as np
import os
from os.path import join as pjoin
from Quaternions import Quaternions
import re
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_3d_motion
import random
import math
import statistics
import torch
import bisect
import re
from data_loaders.truebones.truebones_utils.param_utils import HML_AVG_BONELEN, FOOT_CONTACT_HEIGHT_THRESH, FACE_JOINTS, DATASET_DIR, MAX_PATH_LEN, ANIMATIONS_DIR, MOTION_DIR, NO_BVHS, FOOT_CONTACT_VEL_THRESH, RAW_DATA_DIR, BVHS_DIR
from utils.rotation_conversions import rotation_6d_to_matrix_np


HUMAN_3D_POSE = [7,  0,  1,  2,  0,  4,  5,  8, -1,  8,  9, 10,  8, 12, 13,  0, 15]
HUMAN_3D_OFFSET = np.array([[ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        , -0.10000001,  0.        ],
       [ 0.        , -0.10000001,  0.        ],
       [ 0.        , -0.10000001,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.10000001,  0.        ],
       [ 0.        ,  0.10000001,  0.        ],
       [ 0.        ,  0.10000001,  0.        ],
       [ 0.        ,  0.10000001,  0.        ],
       [ 0.        ,  0.10000001,  0.        ],
       [ 0.        ,  0.10000001,  0.        ],
       [ 0.        ,  0.10000001,  0.        ],
       [ 0.        , -0.20000002,  0.        ]])

t_pose_anim, _1, _2 = animation_from_positions(positions=t_pose_positions, parents=HUMAN_3D_POSE, offsets=HUMAN_3D_OFFSET, iterations=150)