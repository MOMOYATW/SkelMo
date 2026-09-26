from data_loaders.truebones.truebones_utils.motion_process import process_new_object
from utils.parser_util import process_new_skeleton_args

"""
We provide a preprocessing code for skeletons outside the Truebones dataset.
While designed to be as generic as possible, some skeleton-specific adjustments may be needed since it
was originally tailored for Truebones. For example, it relies on joint names for foot classification
and specific velocity/height thresholds for foot contact detection. However, we have tested it on BVH
files from Mixamo and other sources to ensure its generalizability.

Input Arguments:
object_name - A character's indicative name (e.g., "Dog").
bvh_dir - Directory containing BVH files of the skeleton. More files improve statistical accuracy for motion denormalization.
face_joints_names - Four joints defining skeleton orientation ([right hip, left hip, right shoulder, left shoulder] or equivalent).
            Used to align the skeleton to Z+ and XZ plane. Accepts joints names rather than indices since joints indices
            might change at loading.
save_dir - Output directory.
tpos_bvh - A BVH file of the character's natural rest pose for meaningful rotation learning.
        If missing, the code selects a pose from the provided BVH files.

Output:
The code will create the following under save_dir:
save_dir/
        |_motions
        |_animations
        |_bvhs
        cond.npy
1. In motions directory, you will find npy files, which are the processed motion features of each bvh file.
This is useful in case you would like to use this data for training.
2. In animation directory, you will find mp4 files corresponding to each of the processed bvhs.
This is a good sanity check that everything worked as expected.
Note that face_joints are marked in blue and feet joints are marked in green.
3.In bvhs dir you can find the processed bvhs
4. cond.npy contains the skeletons representation, including joints names ambeddings and graph conditions,
which is given as input to SkelMo during inference. Please follow the sampling instructions in README.
"""
def main():
    args = process_new_skeleton_args()
    process_new_object(args.object_name, args.bvh_dir, args.face_joints_names, args.save_dir, args.tpos_bvh)

if __name__ == '__main__':
        main()
