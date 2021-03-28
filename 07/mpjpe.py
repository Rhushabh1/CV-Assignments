# function that calculates mjpje between a batch of poses. 
# For our experiments one of them should be the GT poses 
# and the other be their corresponding predicted ones.
# Note: the function expects numpy arrays and NOT Tensors.
# of the form (batch_size, 15*3) or (batch_size, 15, 3)

def cal_mpjpe(pose_1, pose_2, avg=True):
    n_joints = pose_1.shape[1]
    batch_size = pose_1.shape[0]
    pose_1 = np.copy(pose_1).reshape(batch_size, n_joints, 3)
    pose_2 = np.copy(pose_2).reshape(batch_size, n_joints, 3)
    diff = pose_1-pose_2
    diff_sq = diff ** 2
    dist_per_joint = np.sqrt(np.sum(diff_sq, axis=2))
    dist_per_sample = np.average(dist_per_joint, axis=1)
    if avg is True:
        dist_avg = np.average(dist_per_sample)
    else:
        dist_avg = dist_per_sample
    return dist_avg
