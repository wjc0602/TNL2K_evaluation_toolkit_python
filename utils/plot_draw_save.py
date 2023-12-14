import matplotlib.pyplot as plt
import os
import numpy as np


def plot_draw_save(num_tracker, plot_style, ave_success_rate_plot, idx_seq_set, rank_num,
                   ranking_type, rank_idx, name_tracker_all, threshold_set, title_name,
                   x_label_name, y_label_name, fig_name, save_fig_path, save_fig_suf):
    perf = np.zeros(num_tracker)
    for i in range(num_tracker):
        tmp = ave_success_rate_plot[i, idx_seq_set, :]
        aa = tmp.reshape((len(idx_seq_set), ave_success_rate_plot.shape[2]))
        aa = aa[np.sum(aa, axis=1) > np.finfo(float).eps, :]
        bb = np.mean(aa, axis=0)

        if ranking_type == 'AUC':
            perf[i] = np.mean(bb)
        elif ranking_type == 'threshold':
            perf[i] = bb[rank_idx]

    index_sort = np.argsort(perf)[::-1]

    font_size = 14
    font_size_legend = 14
    axex_font_size = 14

    plt.figure()
    plt.gcf().set_size_inches(10, 6)

    for k in index_sort[:rank_num]:
        tmp = ave_success_rate_plot[k, idx_seq_set, :]
        aa = tmp.reshape((len(idx_seq_set), ave_success_rate_plot.shape[2]))
        aa = aa[np.sum(aa, axis=1) > np.finfo(float).eps, :]
        bb = np.mean(aa, axis=0)

        score = None
        if ranking_type == 'AUC':
            score = np.mean(bb)
        elif ranking_type == 'threshold':
            score = bb[rank_idx]

        tmp_name = f'[{score:.3f}] {name_tracker_all[k]}'
        plt.plot(threshold_set, bb, color=plot_style[k]['color'], linestyle=plot_style[k]['lineStyle'],
                 linewidth=4, label=tmp_name)
        plt.grid(True)
        if k == index_sort[0]:
            plt.grid(linestyle=':', color='k', alpha=1, linewidth=1.2)

    legend_position = 'upper left' if ranking_type == 'threshold' else 'upper right'
    plt.legend(fontsize=font_size_legend, loc=legend_position)
    plt.title(title_name, fontsize=font_size)
    plt.xlabel(x_label_name, fontsize=font_size)
    plt.ylabel(y_label_name, fontsize=font_size)

    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)

    if save_fig_suf == 'eps':
        plt.savefig(os.path.join(save_fig_path, f'{fig_name}.{save_fig_suf}'), format='eps')
    else:
        plt.savefig(os.path.join(save_fig_path, f'{fig_name}.{save_fig_suf}'), format='png')

    plt.close()
