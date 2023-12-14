import numpy as np
from .calc_rect_int import calc_rect_int


def calc_seq_err_robust(results, rect_anno, absent_anno, norm_dst):
    seq_length = rect_anno.shape[0]

    if results.shape[0] != rect_anno.shape[0]:
        results = results[:seq_length, :]

    for i in range(1, seq_length):
        r = results[i, :]  # track result
        r_anno = rect_anno[i, :]  # anno

        if (np.sum(np.isnan(r)) or not np.isreal(r).all() or r[2] <= 0 or r[3] <= 0) and (not np.isnan(r_anno).all()):
            results[i, :] = results[i - 1, :]

    rect_mat = results.copy()
    rect_mat[0, :] = rect_anno[0, :]
    # 删除目标消失的帧不与计算
    absent_idx = absent_anno == 1
    rect_mat = np.delete(rect_mat, np.where(absent_idx), axis=0)
    rect_anno = np.delete(rect_anno, np.where(absent_idx), axis=0)

    center_GT = np.column_stack([(rect_anno[:, 0] + (rect_anno[:, 2] - 1) / 2), (rect_anno[:, 1] + (rect_anno[:, 3] - 1) / 2)])  # GT
    center = np.column_stack([(rect_mat[:, 0] + (rect_mat[:, 2] - 1) / 2), (rect_mat[:, 1] + (rect_mat[:, 3] - 1) / 2)])  # Result

    new_seq_length = rect_anno.shape[0]

    if norm_dst:
        center[:, 0] = center[:, 0] / rect_anno[:, 2]
        center[:, 1] = center[:, 1] / rect_anno[:, 3]
        center_GT[:, 0] = center_GT[:, 0] / rect_anno[:, 2]
        center_GT[:, 1] = center_GT[:, 1] / rect_anno[:, 3]

    err_center = np.sqrt(np.sum(((center[:new_seq_length, :] - center_GT[:new_seq_length, :]) ** 2), axis=1))

    # The officially provided code removes all the bounding boxes in the upper left corner of the border
    # index = rect_anno > 0
    # idx = (np.sum(index, axis=1) == 4)

    # 2023.12.14 Modify to remove only the bounding box with a length and width of 0
    index = rect_anno[:, 2:] > 0
    idx = (np.sum(index, axis=1) == 2)

    tmp = calc_rect_int(rect_mat[idx, :], rect_anno[idx, :])

    errCoverage = -np.ones(len(idx))
    errCoverage[idx] = tmp
    err_center[~idx] = -1

    return errCoverage, err_center
