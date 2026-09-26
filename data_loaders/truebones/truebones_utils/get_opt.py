from argparse import Namespace
import re
from os.path import join as pjoin
from data_loaders.truebones.truebones_utils.param_utils import MAX_JOINTS, FEATS_LEN, MAX_PATH_LEN, FPS, OBJECT_SUBSETS_DICT, DATASET_DIR


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(device, data_root=None):
    opt = Namespace()
    opt.data_root = data_root or DATASET_DIR
    opt.cond_file = pjoin(opt.data_root, 'cond.npy')
    opt.motion_dir = pjoin(opt.data_root, 'motions')
    opt.max_motion_length = 40
    opt.max_joints = MAX_JOINTS
    opt.feature_len = FEATS_LEN
    opt.is_continue = False
    opt.device = device
    opt.max_path_len=MAX_PATH_LEN
    opt.fps=FPS
    opt.subsets_dict=OBJECT_SUBSETS_DICT
    return opt
