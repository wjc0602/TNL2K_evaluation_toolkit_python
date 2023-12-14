import os
import numpy as np
from utils.calc_seq_err_robust import calc_seq_err_robust


def eval_tracker(seqs, trackers, eval_type, name_tracker_all, tmp_mat_path, path_anno, rp_all, norm_dst):
    num_tracker = len(trackers)
    num_seqs = len(seqs)

    threshold_set_overlap = np.arange(0, 1.05, 0.05)
    threshold_set_error = np.arange(0, 51, 1)

    ave_success_rate_plot = np.zeros((num_tracker, num_seqs, len(threshold_set_overlap)))
    ave_success_rate_plot_err = np.zeros((num_tracker, num_seqs, len(threshold_set_error)))

    if norm_dst:
        threshold_set_error = threshold_set_error / 100

    for i, s in enumerate(seqs):  # for each sequence
        anno = np.loadtxt(os.path.join(path_anno, 'gt_rect', f'{s}.txt'), delimiter=',')
        absent_anno = np.loadtxt(os.path.join(path_anno, 'absent', f'{s}.txt'))

        for k, t in enumerate(trackers):  # evaluate each tracker
            try:
                res = np.loadtxt(os.path.join(rp_all, f'{t["name"]}_tracking_result', f'{s}.txt'))
            except FileNotFoundError:
                if t["name"].split('_')[0] == 'mdnet-multiAgent-LG-20191203':
                    res = np.loadtxt(os.path.join(rp_all, t["name"], f'{s}_mdnet-multiAgent-LG-20191203.txt'))
                else:
                    try:
                        res = np.loadtxt(os.path.join(rp_all, t["name"], f'{s}.txt'), delimiter=',')
                    except:
                        res = np.loadtxt(os.path.join(rp_all, t["name"], f'{s}.txt'), delimiter='\t')

            if res.size == 0:
                break

            print(f'evaluating {t["name"]} on {s} ...')

            success_num_overlap = np.zeros(len(threshold_set_overlap))
            success_num_err = np.zeros(len(threshold_set_error))

            err_coverage, err_center = calc_seq_err_robust(res, anno, absent_anno, norm_dst)

            for t_idx, threshold in enumerate(threshold_set_overlap):
                success_num_overlap[t_idx] = np.sum(err_coverage > threshold)

            for t_idx, threshold in enumerate(threshold_set_error):
                success_num_err[t_idx] = np.sum(err_center <= threshold)

            len_all = anno.shape[0]  # number of frames in the sequence

            ave_success_rate_plot[k, i, :] = success_num_overlap / (len_all + np.finfo(float).eps)
            ave_success_rate_plot_err[k, i, :] = success_num_err / (len_all + np.finfo(float).eps)

    # save results
    if not os.path.exists(tmp_mat_path):
        os.makedirs(tmp_mat_path)

    dataName1 = os.path.join(tmp_mat_path, f'aveSuccessRatePlot_{num_tracker}alg_overlap_{eval_type}.npz')
    np.savez(dataName1, ave_success_rate_plot=ave_success_rate_plot, name_tracker_all=name_tracker_all)

    dataName2 = os.path.join(tmp_mat_path, f'aveSuccessRatePlot_{num_tracker}alg_error_{eval_type}.npz')
    ave_success_rate_plot = ave_success_rate_plot_err
    np.savez(dataName2, ave_success_rate_plot=ave_success_rate_plot, name_tracker_all=name_tracker_all)
