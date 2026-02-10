"""
Utility and analysis functions for TSPTW/VRP similarity and visualization.

How to run this file as a script:

1) TSPTW100 Car-POMO feasibility statistics
   python similarity.py --mode tw100_penalty --device cuda

2) TSPTW50 search trajectories (CaR vs NeuOpt) for selected instances
   python similarity.py --mode tw50_traj --instances 10,49,20,80

3) TSPTW50 grid visualization (4x5 solutions of one instance)
   python similarity.py --mode tw50_grid --grid_instance 0 --grid_num 20

4) TSPTW50 similarity metrics vs LKH for multiple methods
   python similarity.py --mode tw50_similarity

By default, --mode is "tw50_traj". All paths used in these analyses are hard-coded
"""

import pickle
import torch
import matplotlib as mpl

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm          # Colormap for generating multiple blue shades

def _ensure_list(arr):
    """Wrap a single item into a list to keep backward compatibility."""
    return arr if isinstance(arr, (list, tuple)) else [arr]

def plot_cvrp_trajectory0(path_lengths_1, feasibility_1,
                         T1=None,
                         reference_value=None, reference_label="Reference Method",
                         method1_name="Method 1",
                         title="CVRP Search Trajectory",
                         save_path=None, base_size=23, wo_legend=False):
    # ---------- 1. Pre-process input ----------
    path_lengths_1 = _ensure_list(path_lengths_1)
    feasibility_1  = _ensure_list(feasibility_1)

    # ---------- 2. Font settings (scaled by base_size) ----------
    plt.rcParams.update({
        'font.size': base_size,
        'axes.titlesize': base_size + 2,
        'axes.labelsize': base_size,
        'xtick.labelsize': base_size - 2,
        'ytick.labelsize': base_size - 2,
        'legend.fontsize': base_size - 1,
        'figure.titlesize': base_size + 4,
        # 'font.family': 'Times New Roman',
    })

    # ---------- 3. Colors ----------
    # Blue gradient: later curves use different blue shades
    n_runs = len(path_lengths_1)
    blues  = cm.get_cmap('Blues', n_runs + 2)
    green  = '#2ca02c'
    refcol = '#8c564b'

    fig, ax = plt.subplots(figsize=(12, 8))

    legend_elems, legend_labels = [], []

    # ---------- 4. Plot each trajectory ----------
    for run_idx, (paths, feas) in enumerate(zip(path_lengths_1, feasibility_1)):
        if T1 is not None:
            paths = paths[:T1+1]
            feas  = feas [:T1+1]

        T_cur      = len(paths)
        iters      = np.arange(1, T_cur+1)
        line_color = blues(run_idx + 2)      # skip the lightest two colors

        # Main polyline
        ax.plot(iters, paths, color=line_color, lw=2.0,
                label=f"{method1_name}-run{run_idx+1}")

        # Feasible / infeasible markers
        for i in range(T_cur):
            marker, mcolor = ('o', green) if feas[i] else ('x', 'red')
            ax.plot(iters[i], paths[i], marker, color=mcolor, ms=7)

        # Best-so-far curve (green dashed line)
        best = np.inf * np.ones_like(paths, dtype=float)
        for i in range(T_cur):
            if feas[i]:
                best[i] = paths[i] if i == 0 else min(best[i-1], paths[i])
            elif i > 0:
                best[i] = best[i-1]
        best[best == np.inf] = np.nan
        ax.plot(iters, best, '--', color=green, lw=2.0)

        # Add legend entries only for the first trajectory
        if run_idx == 0:
            legend_elems.extend([
                Line2D([0], [0], color=line_color, lw=2.0),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=green, markersize=10),
                Line2D([0], [0], marker='x', color='red', markersize=10),
                Line2D([0], [0], color=green, linestyle='--', lw=2.0)
            ])
            legend_labels.extend([
                method1_name,
                'Feasible Solution',
                'Infeasible Solution',
                f'Best-so-far ({method1_name})'
            ])

    # ---------- 5. Reference horizontal line (if provided) ----------
    if reference_value is not None:
        ax.axhline(y=reference_value, color=refcol, lw=2.5)
        legend_elems.append(Line2D([0], [0], color=refcol, lw=2.5))
        legend_labels.append(reference_label)

    # ---------- 6. Axes, title, grid, legend ----------
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Path Length')
    ax.set_title(title)
    ax.grid(ls='--', alpha=0.7)
    if not wo_legend: ax.legend(legend_elems, legend_labels, loc='best')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_cvrp_trajectory000(
        path_lengths_1, feasibility_1,
        path_lengths_2=None, feasibility_2=None,
        path_lengths_3=None, feasibility_3=None,
        T1=None, T2=None, T3=None,
        reference_value=None, reference_label="Reference Method",
        method1_name="Method 1", method2_name="Method 2", method3_name="Method 3",
        title="CVRP Search Trajectory", save_path=None,
        base_size=28                     # unified base font size
):
    # ---------- 1. Global bold font + sizes ----------
    plt.rcParams.update({
        'font.size'       : base_size,
        'font.weight'     : 'bold',      # global bold
        'axes.titlesize'  : base_size + 4,
        'axes.labelsize'  : base_size + 2,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'figure.titlesize': base_size + 6,
        'xtick.labelsize' : base_size,
        'ytick.labelsize' : base_size,
        'legend.fontsize' : base_size - 3,
    })

    # ---------- 2. Optional time truncation ----------
    if T1 is not None:
        path_lengths_1, feasibility_1 = path_lengths_1[:T1+1], feasibility_1[:T1+1]
    if path_lengths_2 is not None and T2 is not None:
        path_lengths_2, feasibility_2 = path_lengths_2[:T2+1], feasibility_2[:T2+1]
    if path_lengths_3 is not None and T3 is not None:
        path_lengths_3, feasibility_3 = path_lengths_3[:T3+1], feasibility_3[:T3+1]

    # ---------- 3. Figure & colors ----------
    plt.figure(figsize=(12, 8))
    colors = {
        'line'      : ['#1f77b4', '#9467bd', '#d62728'],
        'best_sofar': ['#2ca02c', '#ff7f0e', '#e377c2'],
        'ref'       : '#8c0000'
    }

    def best_feasible(vals, feas):
        out = np.full_like(vals, np.inf, dtype=float)
        best = np.inf
        for i, (v, f) in enumerate(zip(vals, feas)):
            if f:
                best = min(best, v)
            out[i] = best
        out[out == np.inf] = np.nan
        return out

    # ---------- 4. Plot each method ----------
    for idx, (pl, fea, name) in enumerate([
        (path_lengths_1, feasibility_1, method1_name),
        (path_lengths_2, feasibility_2, method2_name) if path_lengths_2 is not None else (None, None, None),
        (path_lengths_3, feasibility_3, method3_name) if path_lengths_3 is not None else (None, None, None),
    ]):
        if pl is None:    # skip empty method
            continue
        its = np.arange(1, len(pl) + 1)

        # Main polyline
        plt.plot(its, pl, color=colors['line'][idx], lw=2.5, label=name)

        # Feasible points: green circles; infeasible points: red crosses
        feas_idx  = np.where(fea)[0]
        infeas_idx = np.where(~np.array(fea))[0]
        if feas_idx.size:
            plt.scatter(its[feas_idx], np.array(pl)[feas_idx],
                        marker='o', c='g', s=90, linewidths=0)
        if infeas_idx.size:
            plt.scatter(its[infeas_idx], np.array(pl)[infeas_idx],
                        marker='x', c='r', s=120, linewidths=3)

        # Best-so-far feasible curve
        plt.plot(its, best_feasible(pl, fea),
                 color=colors['best_sofar'][idx], ls='--', lw=2.5,
                 label=f'Best-so-far ({name})')

    # ---------- 5. Reference line ----------
    if reference_value is not None:
        plt.axhline(reference_value, color=colors['ref'], lw=3,
                    label=reference_label)

    # ---------- 6. Final details ----------
    # plt.ylim([25.8,27.6])
    plt.xlabel('Iteration')
    plt.ylabel('Obj.')
    from matplotlib.ticker import FormatStrFormatter
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.title(title)
    plt.grid(ls='--', alpha=.7)
    plt.tight_layout()
    plt.legend()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')

    plt.show()


def visualize_tsp_with_times_new_roman(route, coordinates, feasibility):
    # Set global font to Times New Roman
    mpl.rcParams['font.family'] = 'Times New Roman'

    xs = [coordinates[node][0] for node in route]
    ys = [coordinates[node][1] for node in route]

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, linestyle='-', color='black', linewidth=1)

    for node in route:
        x, y = coordinates[node]
        node_color = 'red' if not feasibility[node] else 'green'
        plt.scatter(x, y, color=node_color, s=200, edgecolors='black', marker='o', zorder=5)
        plt.text(x + 0.02, y + 0.02, f'{node}', fontsize=14)

    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], linestyle='-', color='black', linewidth=1)

    for t in [0, 0.5, 1]:
        plt.text(t, -0.03, f'{t:.1f}', ha='center', va='top', fontsize=14)
        plt.text(-0.03, t, f'{t:.1f}', ha='right', va='center', fontsize=14)

    plt.title("TSP Solution in Unit Square", fontsize=16)
    plt.tight_layout()
    plt.show()


def visualize(route, coordinates, feasibility, method_name="", tour_length=None):
    # mpl.rcParams['font.family'] = 'Times New Roman'

    xs = [coordinates[node][0] for node in route]
    ys = [coordinates[node][1] for node in route]

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, linestyle='-', color='black', linewidth=1)

    for node in route:
        x, y = coordinates[node]
        node_color = 'red' if not feasibility[node] else 'green'
        plt.scatter(x, y, color=node_color, s=200, edgecolors='black', marker='o', zorder=5)
        # plt.text(x + 0.02, y + 0.02, f'{node}', fontsize=14)

    ax = plt.gca()
    # Set axis range to [-0.02, 1.02]
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal', adjustable='box')

    # Hide default axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Draw a clearer outer box
    box_x = [-0.02, 1.02, 1.02, -0.02, -0.02]
    box_y = [-0.02, -0.02, 1.02, 1.02, -0.02]
    plt.plot(box_x, box_y, linestyle='-', color='black', linewidth=1.5)  # 增加线宽

    # Emphasize bottom and right borders
    plt.plot([1.02, -0.02], [-0.02, -0.02], color='black', linewidth=2.0)  # 底部边框
    plt.plot([1.02, 1.02], [-0.02, 1.02], color='black', linewidth=2.0)  # 右侧边框

    # Only mark ticks at 0 and 1 with larger bold font
    for t in [0, 1]:
        plt.text(t, -0.05, f'{t:.1f}', ha='center', va='top', fontsize=14)
        plt.text(-0.05, t, f'{t:.1f}', ha='right', va='center', fontsize=14)

    # Title includes tour-length information
    title = "TSPTW-50 Solution"
    if method_name:
        title += f" ({method_name})"
    if tour_length is not None and (feasibility.all()):
        title += f"\nTour length: {tour_length:.3f}"
    else:
        title += f"\nInfeasible"

    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"tsptw_50_solution_{method_name}_80_aug1.png", bbox_inches='tight')
    plt.show()

def visualize_subplot(ax, route, coordinates, feasibility, method_name="", tour_length=None):
    xs = [coordinates[node][0] for node in route]
    ys = [coordinates[node][1] for node in route]

    ax.plot(xs, ys, linestyle='-', color='black', linewidth=1)

    for node in route:
        x, y = coordinates[node]
        node_color = 'red' if not feasibility[node] else 'green'
        ax.scatter(x, y, color=node_color, s=60, edgecolors='black', marker='o', zorder=5)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal', adjustable='box')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # 外边框
    box_x = [-0.02, 1.02, 1.02, -0.02, -0.02]
    box_y = [-0.02, -0.02, 1.02, 1.02, -0.02]
    ax.plot(box_x, box_y, linestyle='-', color='black', linewidth=1.5)
    ax.plot([1.02, -0.02], [-0.02, -0.02], color='black', linewidth=2.0)
    ax.plot([1.02, 1.02], [-0.02, 1.02], color='black', linewidth=2.0)

    for t in [0, 1]:
        ax.text(t, -0.05, f'{t:.1f}', ha='center', va='top', fontsize=8)
        ax.text(-0.05, t, f'{t:.1f}', ha='right', va='center', fontsize=8)

    # 小标题
    title = f"T = {method_name}"
    if tour_length is not None and feasibility.all():
        title += f"\nLen: {tour_length:.4f}"
    else:
        title += f"\nInfeasible"
    ax.set_title(title, fontsize=18, linespacing=1.2)


def hamming_distance(route1, route2):
    """
    Compute the Hamming distance between two routes: number of differing positions.
    """
    if len(route1) != len(route2):
        raise ValueError("Two routes must have the same length.")

    return sum(1 for i in range(len(route1)) if route1[i] != route2[i])


def levenshtein_distance(route1, route2):
    """
    Compute the edit (Levenshtein) distance between two routes:
    minimum number of operations (insert, delete, replace) to transform one into the other.
    """
    n = len(route1)
    m = len(route2)

    # Create DP matrix
    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    # Initialize boundaries
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if route1[i - 1] == route2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],  # 删除
                                   dp[i][j - 1],  # 插入
                                   dp[i - 1][j - 1])  # 替换

    return dp[n][m]


def kendall_tau_distance(route1, route2):
    """
    Compute Kendall tau distance between two routes: inconsistency between two permutations.
    """
    if len(route1) != len(route2):
        raise ValueError("Two routes must have the same length.")

    # Map element -> position in route1
    pos1 = {route1[i]: i for i in range(len(route1))}

    # Count inversions
    inversions = 0
    for i in range(len(route2)):
        for j in range(i + 1, len(route2)):
            # Inversion if the relative order differs between route1 and route2
            if pos1[route2[i]] > pos1[route2[j]]:
                inversions += 1

    return inversions


def lcs_length(route1, route2):
    """
    Compute the length of the Longest Common Subsequence (LCS) between two routes.
    """
    n = len(route1)
    m = len(route2)

    # Create DP matrix
    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if route1[i - 1] == route2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[n][m]


def get_lcs(route1, route2):
    """
    Get the Longest Common Subsequence (LCS) between two routes.
    """
    n = len(route1)
    m = len(route2)

    # Create DP matrix
    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if route1[i - 1] == route2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Recover LCS by backtracking
    lcs = []
    i, j = n, m
    while i > 0 and j > 0:
        if route1[i - 1] == route2[j - 1]:
            lcs.append(route1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs.reverse()

    return lcs


def edge_overlap(route1, route2):
    """
    Compute edge-overlap ratio between two routes:
    number of shared directed edges divided by total edges.
    """
    if len(route1) != len(route2):
        raise ValueError("Two routes must have the same length.")

    # Build edge sets
    edges1 = set()
    edges2 = set()

    for i in range(len(route1)):
        # Add edge (current_node, next_node)
        edges1.add((route1[i], route1[(i + 1) % len(route1)]))
        edges2.add((route2[i], route2[(i + 1) % len(route2)]))

    # Shared edges
    common_edges = edges1.intersection(edges2)

    # Overlap ratio
    overlap_ratio = len(common_edges) / len(edges1)

    return overlap_ratio


def tsp_similarity(route1, route2):
    """
    Compute a collection of TSP route similarity metrics between two routes.
    Returns a dictionary containing all distance and similarity scores.
    """
    # Distances / similarities
    hamming = hamming_distance(route1, route2)
    edit = levenshtein_distance(route1, route2)
    kendall = kendall_tau_distance(route1, route2)
    lcs_len = lcs_length(route1, route2)
    lcs = get_lcs(route1, route2)
    overlap = edge_overlap(route1, route2)

    # Normalized similarity scores (higher means more similar)
    n = len(route1)
    hamming_sim = 1 - hamming / n
    edit_sim = 1 - edit / (2 * n)  # 最大编辑距离约为2n
    kendall_sim = 1 - kendall / (n * (n - 1) / 2)  # 最大逆序对数量
    lcs_sim = lcs_len / n

    # Pack results
    results = {
        "hamming_distance": hamming,
        "hamming_similarity": hamming_sim,
        "edit_distance": edit,
        "edit_similarity": edit_sim,
        "kendall_tau_distance": kendall,
        "kendall_tau_similarity": kendall_sim,
        "lcs_length": lcs_len,
        "lcs": lcs,
        "lcs_similarity": lcs_sim,
        "edge_overlap_ratio": overlap
    }

    return results


def load(filename, offset=0, num_samples=10000):
    with open(filename, 'rb') as f:
        data = pickle.load(f)[offset: offset + num_samples]
    print(">> Load {} data ({}) from {}".format(len(data), type(data), filename))
    return data

def cal_gap(dpdp, lkh):
    with open(dpdp, 'rb') as file:
        dpdp = pickle.load(file)
    with open(lkh, 'rb') as file:
        lkh = pickle.load(file)

    gaps = []
    obj = []
    infeasible_cnt = 0
    for i in range(10000):
        gap = (dpdp[i][0] - lkh[i][0]) / lkh[i][0] *100
        if (dpdp[i][1] == np.arange(1, len(dpdp[0][1])+1)).all():
            print(i)
            infeasible_cnt += 1
            # infsb
        else:
            # feasible
            gaps.append(gap)
            obj.append(dpdp[i][0])
        # if dpdp[0][i][0] is not None:
        #     gap = (dpdp[0][i][0] - lkh[i][0]) / lkh[i][0]
        #     gaps.append(gap)
    print("Infeasible rate: ", infeasible_cnt / 10000 * 100)
    print("Gap: ", np.mean(gaps))
    print("Obj: ", np.mean(obj))

def _get_travel_distance(solution, node_xy):
    gathering_index = solution[:, :, :, None].expand(-1, -1, -1, 2)
    # shape: (batch, pomo, selected_list_length, 2)
    all_xy = node_xy[:, None, :, :].expand(-1, 1, -1, -1)
    # shape: (batch, pomo, problem+1, 2)

    ordered_seq = all_xy.gather(dim=2, index=gathering_index)
    # shape: (batch, pomo, selected_list_length, 2)

    rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
    segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
    # shape: (batch, pomo, selected_list_length)

    travel_distances = segment_lengths.sum(2)
    # shape: (batch, pomo)
    return travel_distances/100


def calculate_penalty(solution, raw_node_xy, raw_tw_start, raw_tw_end):
    """
    Calculate time window constraint penalties for TSPTW problem.

    Args:
        solution: Tensor of shape [batch_size, pomo_size, solution_size] or [batch_size*pomo_size, solution_size]
                 Contains node indices for each solution.
        raw_node_xy: Tensor of shape [batch_size, num_nodes, 2] with node coordinates.
        raw_tw_start: Tensor of shape [batch_size, num_nodes] with time window start times.
        raw_tw_end: Tensor of shape [batch_size, num_nodes] with time window end times.

    Returns:
        timeouts: Actual timeout values at each node (how much the time window was violated).
        node_timeouts: Binary indicators of whether time windows were violated at each node.
    """
    # Handle input dimensions
    batch_size = raw_node_xy.size(0)
    if solution.dim() == 3:
        # Solution already has [batch_size, pomo_size, solution_size] shape
        _, pomo_size, solution_size = solution.size()
        solution = solution.view(batch_size * pomo_size, -1)
    else:
        # Solution has [batch_size*pomo_size, solution_size] shape
        solution_size = solution.size(-1)
        pomo_size = solution.size(0) // batch_size

    # Create batch indices for gathering
    batch_indices = torch.arange(batch_size * pomo_size, device=solution.device).unsqueeze(1)

    # Calculate distance matrix and replicate for each POMO (parallel optimization multi-operator)
    raw_dist = torch.cdist(raw_node_xy, raw_node_xy, p=2)
    raw_dist = raw_dist.repeat_interleave(pomo_size, dim=0)

    # Get travel times between consecutive nodes in each solution
    travel_time_all = raw_dist[batch_indices, solution[:, :-1], solution[:, 1:]]

    # Replicate time window constraints for each POMO
    raw_tw_start = raw_tw_start.repeat_interleave(pomo_size, dim=0)
    raw_tw_end = raw_tw_end.repeat_interleave(pomo_size, dim=0)

    # Pre-allocate tensors to avoid concatenation in loop
    current_time = torch.zeros(batch_size * pomo_size, device=solution.device)
    timeouts = torch.zeros(batch_size * pomo_size, solution_size, device=solution.device)

    # Calculate arrival times and timeouts for each node in the solution
    for i in range(solution_size - 1):
        current_node = solution[:, i + 1]
        travel_time = travel_time_all[:, i]

        # Get time window for current node
        tw_start = raw_tw_start[batch_indices, current_node.unsqueeze(-1)].squeeze(-1)
        tw_end = raw_tw_end[batch_indices, current_node.unsqueeze(-1)].squeeze(-1)

        # Update current time (max of arrival time and time window start)
        current_time = torch.max(current_time + travel_time, tw_start)

        # Calculate timeout (how much the time window was violated)
        timeout = torch.clamp(current_time - tw_end, min=0)
        timeouts[:, i + 1] = timeout

    # Convert to binary indicators (0: no violation, 1: violation)
    node_timeouts = (timeouts > 1e-5).int()

    return timeouts, node_timeouts
def calculate_penalty1(solution, raw_node_xy, raw_tw_start, raw_tw_end):

    batch_size = raw_node_xy.size(0)
    solution = solution.view(batch_size, -1, solution.size(-1))
    _, pomo_size, solution_size = solution.size()
    solution = solution.view(batch_size*pomo_size, -1)
    batch_indices = torch.arange(batch_size*pomo_size).unsqueeze(1)

    raw_dist = torch.cdist(raw_node_xy, raw_node_xy, p=2)
    raw_dist = raw_dist.repeat_interleave(pomo_size, dim=0)
    travel_time_all = raw_dist[batch_indices, solution[:, :-1], solution[:, 1:]]
    current_time = torch.zeros(batch_size*pomo_size)
    timeouts = torch.zeros(batch_size*pomo_size, 1)
    raw_tw_start = raw_tw_start.repeat_interleave(pomo_size, dim=0)
    raw_tw_end = raw_tw_end.repeat_interleave(pomo_size, dim=0)
    for i in range(solution_size - 1):
        current_node = solution[:, i + 1]
        travel_time = travel_time_all[:, i]
        tw_start = raw_tw_start[batch_indices, current_node.unsqueeze(-1)].view(-1)
        current_time = torch.max(current_time + travel_time, tw_start)
        # current_time_tensor = torch.cat((current_time_tensor, current_time.unsqueeze(-1)), dim=-1)
        tw_end = raw_tw_end[batch_indices, current_node.unsqueeze(-1)].view(-1)
        timeout = torch.clamp(current_time - tw_end, min=0)
        timeouts = torch.cat((timeouts, timeout.unsqueeze(-1)), dim=-1)
    node_timeouts = torch.where(timeouts > 1e-5, torch.ones_like(timeouts), timeouts).int()

    return timeouts, node_timeouts

def calculate_penalty0(
    solution,           # (B, P, L)
    raw_node_xy,        # (B, N, 2)
    raw_tw_start,       # (B, N)
    raw_tw_end,         # (B, N)
    raw_service_time=None,     # (B, N) ➊ optional
    eps: float = 1e-5,         # ➋ tolerance for floating-point noise
):
    """
    Returns
    -------
    timeouts      : (B·P, L-1)  lateness at every visited node (0 = on time)
    node_timeouts : (B·P, L-1)  binary mask, 1 if late
    """

    # ------------------------------------------------------------------
    # 0. Shapes & common meta
    # ------------------------------------------------------------------
    B, P, L        = solution.size()
    device, dtype  = raw_node_xy.device, raw_node_xy.dtype       # ➌ device / dtype safety
    sol_flat       = solution.reshape(B * P, L)                  # (B·P, L)

    # ------------------------------------------------------------------
    # 1. Distance & travel time for every hop (vectorised)
    # ------------------------------------------------------------------
    dist = torch.cdist(raw_node_xy, raw_node_xy, p=2)            # (B, N, N)
    dist = dist.repeat_interleave(P, dim=0)                      # (B·P, N, N)

    batch_rows = torch.arange(B * P, device=device).unsqueeze(-1)
    travel_time_all = dist[batch_rows, sol_flat[:, :-1], sol_flat[:, 1:]]  # (B·P, L-1)

    # ------------------------------------------------------------------
    # 2. Repeat time windows & (optional) service times along POMO dim
    # ------------------------------------------------------------------
    tw_s = raw_tw_start.repeat_interleave(P, dim=0)              # (B·P, N)
    tw_e = raw_tw_end.repeat_interleave(P, dim=0)                # (B·P, N)
    if raw_service_time is None:
        svc = torch.zeros_like(tw_s)                             # ➍ default 0
    else:
        svc = raw_service_time.repeat_interleave(P, dim=0)       # (B·P, N)

    # ------------------------------------------------------------------
    # 3. Simulate the tour
    # ------------------------------------------------------------------
    current_time = torch.zeros(B * P, device=device, dtype=dtype)
    lateness     = []                                            # collect stepwise

    for step in range(L - 1):
        nxt   = sol_flat[:, step + 1]     # (B·P,)
        travel= travel_time_all[:, step]  # (B·P,)

        # wait until window opens, then add travel time
        earliest = tw_s[batch_rows.squeeze(-1), nxt]
        current_time = torch.maximum(current_time + travel, earliest)

        # check lateness
        latest  = tw_e[batch_rows.squeeze(-1), nxt]
        late    = torch.clamp(current_time - latest, min=0.0)
        lateness.append(late)

        # stay for service time before leaving
        current_time += svc[batch_rows.squeeze(-1), nxt]         # ➎ add service

    # (B·P, L-1)
    timeouts = torch.stack(lateness, dim=1)
    node_timeouts = (timeouts > eps).int()

    return timeouts, node_timeouts

def load_and_process_solution(path, node_xy, tw_start, tw_end):
    """
    Load a solution pickle file, extract routes and costs, compute time-window violations,
    and determine feasibility.

    Args:
        path (str): Path to the .pkl file.
        node_xy (Tensor): Node coordinates.
        tw_start (Tensor): Time window start for each node.
        tw_end (Tensor): Time window end for each node.

    Returns:
        solution (Tensor): shape (B, N) routes.
        cost (Tensor): shape (B,) route costs.
        feasible_mask (Tensor): shape (B,) boolean feasibility mask (time windows satisfied).
    """
    data = load(path)
    solution = torch.tensor([s[2] for s in data])
    fsb = torch.tensor([s[1] for s in data])
    cost = torch.tensor([s[0] for s in data])
    _, node_timeouts = calculate_penalty(solution, raw_node_xy=node_xy, raw_tw_start=tw_start, raw_tw_end=tw_end)
    feasible_mask = (node_timeouts.sum(-1) < 1e-5)

    print(feasible_mask.sum(), (feasible_mask == fsb).sum())
    print(">> Completed solution loading..")

    return solution, fsb, cost, feasible_mask, node_timeouts

def get_travel_distance(solution, node_xy):
    gathering_index = solution[:, :, :, None].expand(-1, -1, -1, 2)
    # shape: (batch, pomo, selected_list_length, 2)
    all_xy = node_xy[:, None, :, :].expand(-1, solution.size(1), -1, -1)
    # shape: (batch, pomo, problem+1, 2)

    ordered_seq = all_xy.gather(dim=2, index=gathering_index)
    # shape: (batch, pomo, selected_list_length, 2)

    rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
    segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
    # shape: (batch, pomo, selected_list_length)

    travel_distances = segment_lengths.sum(2)
    # shape: (batch, pomo)
    return travel_distances


def find_best_solutions_and_trajectories(penalty_history, distance_history, solution_history):
    """
    Find the best solution and its full trajectory for each instance.

    Args:
        penalty_history: (aug, instance, steps)
        distance_history: (aug, instance, steps)
        solution_history: (aug, instance, steps, solution_length)

    Returns:
        best_solutions: (instance, solution_length)
        best_penalties: (instance, steps)
        best_distances: (instance, steps)
        best_solution_trajectories: (instance, steps, solution_length)
        best_indices: (instance, 2) with (aug_idx, step_idx)
    """
    aug, instance, steps = penalty_history.shape
    solution_length = solution_history.shape[-1]

    best_solutions = torch.zeros(
        instance, solution_length, dtype=solution_history.dtype, device=solution_history.device
    )
    best_penalties = torch.zeros(
        instance, steps, dtype=penalty_history.dtype, device=penalty_history.device
    )
    best_distances = torch.zeros(
        instance, steps, dtype=distance_history.dtype, device=distance_history.device
    )
    best_solution_trajectories = torch.zeros(
        instance, steps, solution_length, dtype=solution_history.dtype, device=solution_history.device
    )
    best_indices = torch.zeros(instance, 2, dtype=torch.long)

    for inst in range(instance):
        inst_penalties = penalty_history[:, inst, :]
        inst_distances = distance_history[:, inst, :]
        inst_solutions = solution_history[:, inst, :, :]  # (aug, steps, L)

        flat_penalties = inst_penalties.reshape(-1)
        flat_distances = inst_distances.reshape(-1)

        feasible_mask = (flat_penalties == 0)

        if feasible_mask.any():
            # Prefer feasible solutions with minimal distance
            feasible_distances = flat_distances.clone()
            feasible_distances[~feasible_mask] = float("inf")
            best_flat_idx = torch.argmin(feasible_distances).item()
        else:
            # If no feasible solution, choose one with minimal penalty
            best_flat_idx = torch.argmin(flat_penalties).item()

        best_aug_idx = best_flat_idx // steps
        best_step_idx = best_flat_idx % steps

        best_solutions[inst] = inst_solutions[best_aug_idx, best_step_idx]
        best_penalties[inst] = inst_penalties[best_aug_idx]
        best_distances[inst] = inst_distances[best_aug_idx]
        best_solution_trajectories[inst] = inst_solutions[best_aug_idx]
        best_indices[inst] = torch.tensor([best_aug_idx, best_step_idx])

    return best_solutions, best_penalties, best_distances, best_solution_trajectories, best_indices


# ======================= Analysis / Visualization CLI ========================= #

import os
from typing import List


ROOT = "../"

# TSPTW datasets
TSPTW50_INS = os.path.join(ROOT, "data/TSPTW/tsptw50_da_silva_uniform.pkl")
TSPTW100_INS = os.path.join(ROOT, "data/TSPTW/tsptw100_da_silva_uniform_varyN.pkl")

# LKH solutions
LKH_TSPTW50 = os.path.join(ROOT, "data/TSPTW/lkh_tsptw50_da_silva_uniform.pkl")

# Car-POMO history / solutions
CAR_POMO_HISTORY_SOL = os.path.join(ROOT, "tw100_car_pomo.pt")
CAR_POMO_FSB = os.path.join(ROOT, "fsb_car_pomo.pt")
CAR_POMO_REWARD = os.path.join(ROOT, "rewardd_car_pomo.pt")

# NeuOpt trajectories
NEUOPT_LEN = os.path.join(ROOT, "all_pack/NeuOpt-main/neuopt1000_len.pt")
NEUOPT_FSB = os.path.join(ROOT, "all_pack/NeuOpt-main/neuopt1000_fsb.pt")

# Reference scores (e.g., POMO*)
REFERENCE_SCORE = os.path.join(ROOT, "pomo_aug_score.pt")

# Best-solution pickles for various methods (TSPTW50)
POMO_STAR_BEST = os.path.join(ROOT, "rebut/tw50_pomo_star_best_solution.pkl")
PIP_BEST = os.path.join(ROOT, "rebut/tw50_pip_best_solution.pkl")
NEUOPT_T5000_BEST = os.path.join(ROOT, "rebut/tw50_neuopt_t5000_best_solution.pkl")
NEUOPT_T20_BEST = os.path.join(ROOT, "rebut/tw50_neuopt_t20_best_solution.pkl")
CAR_POMO_T20_BEST = os.path.join(ROOT, "rebut/tw50_car_pomo_t20_best_solution.pkl")
CAR_PIP_T20_BEST = os.path.join(ROOT, "rebut/tw50_car_pip_t20_best_solution.pkl")


def load_tsptw_instance(path: str):
    """Load TSPTW instances and split into tensors (node_xy, service_time, tw_start, tw_end)."""
    data = load(path)
    node_xy = torch.tensor([inst[0] for inst in data], dtype=torch.float32)
    service_time = torch.tensor([inst[1] for inst in data], dtype=torch.float32)
    tw_start = torch.tensor([inst[2] for inst in data], dtype=torch.float32)
    tw_end = torch.tensor([inst[3] for inst in data], dtype=torch.float32)
    return node_xy, service_time, tw_start, tw_end


def analyze_tw100_car_pomo(device: str = "cuda"):
    """
    For TSPTW100:
        - Load dataset and Car-POMO solutions.
        - Compute time-window penalties.
        - Print basic feasibility statistics.
    """
    node_xy, service_time, tw_start, tw_end = load_tsptw_instance(TSPTW100_INS)

    if device == "cuda" and torch.cuda.is_available():
        node_xy = node_xy.cuda()
        service_time = service_time.cuda()
        tw_start = tw_start.cuda()
        tw_end = tw_end.cuda()

    car_pomo_sol = torch.load(CAR_POMO_HISTORY_SOL).squeeze(1)

    repeat_factor = 8  # number of POMO samples per instance
    node_xy_rep = node_xy.repeat_interleave(repeat_factor, dim=0)
    tw_start_rep = tw_start.repeat_interleave(repeat_factor, dim=0)
    tw_end_rep = tw_end.repeat_interleave(repeat_factor, dim=0)

    _, node_timeouts = calculate_penalty(
        car_pomo_sol,
        raw_node_xy=node_xy_rep,
        raw_tw_start=tw_start_rep,
        raw_tw_end=tw_end_rep,
    )

    feasible_mask = (node_timeouts.sum(-1) < 1e-5)
    total = feasible_mask.numel()
    num_feasible = feasible_mask.sum().item()

    print("=== TSPTW100 Car-POMO feasibility ===")
    print(f"Total solutions: {total}")
    print(f"Feasible solutions: {num_feasible}")
    print(f"Feasibility rate: {num_feasible / total * 100:.2f}%")


def plot_tw50_trajectories(selected_instances: List[int] = None, base_size: int = 38):
    """
    For TSPTW50:
        - Load NeuOpt and Car-POMO search trajectories.
        - Plot search trajectories for selected instances and save to PDF.
    This corresponds to the plotting logic in the `opensss` block.
    """
    if selected_instances is None:
        selected_instances = [10, 49, 20, 80]

    reference = torch.load(REFERENCE_SCORE, map_location="cpu")
    car_fsb = torch.load(CAR_POMO_FSB, map_location="cpu").cpu()
    car_reward = torch.load(CAR_POMO_REWARD, map_location="cpu").cpu()
    path_lengths_2 = torch.load(NEUOPT_LEN, map_location="cpu")
    feasibility_2 = torch.load(NEUOPT_FSB, map_location="cpu")

    print("=== Plotting TSPTW50 search trajectories ===")
    for inst_idx in selected_instances:
        ref_value = reference[inst_idx].cpu()
        print(f"Instance {inst_idx}, reference score: {ref_value:.4f}")

        # Car-POMO trajectories (8 augmentation runs)
        for aug_idx in range(8):
            save_path = f"new_plot/{inst_idx}-{aug_idx}-CaR.pdf"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f"  Saving CaR trajectory to {save_path}")
            plot_cvrp_trajectory000(
                car_reward[inst_idx][aug_idx].cpu(),
                car_fsb[inst_idx][aug_idx].cpu(),
                T1=20,
                reference_label="POMO*" if ref_value < 50 else None,
                reference_value=float(ref_value) if ref_value < 50 else None,
                method1_name="CaR",
                title="",
                save_path=save_path,
                base_size=base_size,
            )

        # NeuOpt trajectories
        neuopt_save = f"new_plot/{inst_idx}-NeuOpt.pdf"
        os.makedirs(os.path.dirname(neuopt_save), exist_ok=True)
        print(f"  Saving NeuOpt trajectory to {neuopt_save}")
        plot_cvrp_trajectory000(
            path_lengths_2[inst_idx].cpu(),
            feasibility_2[inst_idx].cpu(),
            T1=20,
            reference_label="POMO*" if ref_value < 50 else None,
            reference_value=float(ref_value) if ref_value < 50 else None,
            method1_name="NeuOpt",
            title="",
            save_path=neuopt_save,
            base_size=base_size,
        )


def visualize_tw50_solutions_grid(instance_idx: int = 0, num_solutions: int = 20):
    """
    Build a 4x5 grid of Car-POMO solutions for a single instance and save as PDF.
    This corresponds to the multi-subplot visualization code in the original script.
    """
    node_xy, service_time, tw_start, tw_end = load_tsptw_instance(TSPTW50_INS)
    car = torch.load(os.path.join(ROOT, "rebut/tw50_car_pomo_history_solution.pt"), map_location="cpu").cpu()

    node_xy_inst = node_xy[instance_idx].unsqueeze(0).repeat_interleave(num_solutions, dim=0)
    tw_start_inst = tw_start[instance_idx].unsqueeze(0).repeat_interleave(num_solutions, dim=0)
    tw_end_inst = tw_end[instance_idx].unsqueeze(0).repeat_interleave(num_solutions, dim=0)

    car_inst = car[0, instance_idx, :num_solutions, :]  # (num_solutions, L)

    travel_dist = get_travel_distance(
        car_inst.unsqueeze(0),            # (B=1, P=num_solutions, L)
        node_xy_inst.unsqueeze(0),        # (B=1, N, 2)
    ).view(-1)
    _, timeouts = calculate_penalty(
        car_inst,
        raw_node_xy=node_xy_inst,
        raw_tw_start=tw_start_inst,
        raw_tw_end=tw_end_inst,
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    for i in range(num_solutions):
        ax = axes[i // 5, i % 5]
        route = list(car_inst[i].numpy())
        coords = list((node_xy_inst[i] / 100).numpy())
        feas = np.full(len(route), (timeouts[i] < 1e-5).all().item())
        visualize_subplot(
            ax,
            route,
            coords,
            feas,
            method_name=str(i),
            tour_length=travel_dist[i].item(),
        )

    plt.tight_layout()
    save_path = f"all_{num_solutions}_tsptw_solutions_inst{instance_idx}.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"Saved grid visualization to {save_path}")


def compute_tw50_similarity_stats():
    """
    Compute average similarity metrics between multiple methods and LKH solutions on TSPTW50.

    This is a cleaned-up version of the large commented similarity block in the original script.
    """
    node_xy, service_time, tw_start, tw_end = load_tsptw_instance(TSPTW50_INS)

    lkh_raw = load(LKH_TSPTW50)
    lkh_routes = torch.cat(
        [torch.zeros((len(lkh_raw), 1), dtype=torch.int32), torch.tensor([s[1] for s in lkh_raw], dtype=torch.int32)],
        dim=-1,
    )

    _, lkh_timeouts = calculate_penalty(
        lkh_routes,
        raw_node_xy=node_xy,
        raw_tw_start=tw_start,
        raw_tw_end=tw_end,
    )
    lkh_feasible = (lkh_timeouts.sum(-1) < 1e-5)

    pomo_star_solution, pomo_star_fsb, pomo_star_cost, pomo_star_feasible, pomo_star_timeouts = load_and_process_solution(
        POMO_STAR_BEST, node_xy, tw_start, tw_end
    )
    pip_solution, pip_fsb, pip_cost, pip_feasible, pip_timeouts = load_and_process_solution(
        PIP_BEST, node_xy, tw_start, tw_end
    )
    refine_solution, refine_fsb, refine_cost, refine_feasible, refine_timeouts = load_and_process_solution(
        NEUOPT_T20_BEST, node_xy, tw_start, tw_end
    )
    impr_solution, impr_fsb, impr_cost, impr_feasible, impr_timeouts = load_and_process_solution(
        NEUOPT_T5000_BEST, node_xy, tw_start, tw_end
    )
    car_pomo_solution, car_pomo_fsb, car_pomo_cost, car_pomo_feasible, car_pomo_timeouts = load_and_process_solution(
        CAR_POMO_T20_BEST, node_xy, tw_start, tw_end
    )
    car_pip_solution, car_pip_fsb, car_pip_cost, car_pip_feasible, car_pip_timeouts = load_and_process_solution(
        CAR_PIP_T20_BEST, node_xy, tw_start, tw_end
    )

    fsb_list = [pomo_star_feasible, pip_feasible, refine_feasible, impr_feasible, car_pomo_feasible, car_pip_feasible]
    methods = ["POMO*", "PIP", "Refine", "Improvement", "Car-POMO", "Car-PIP"]
    solution_list = [pomo_star_solution, pip_solution, refine_solution, impr_solution, car_pomo_solution, car_pip_solution]

    print("=== Similarity statistics vs LKH on TSPTW50 ===")

    for i, method_name in enumerate(methods):
        sols = solution_list[i]
        fsb = fsb_list[i]

        print(f"\n==== {method_name} ====")

        valid_count = 0
        hamming_sum = levenshtein_sum = lcs_sum = edge_sum = kendall_sum = 0.0

        for j in range(len(sols)):
            if fsb[j] and lkh_feasible[j]:
                valid_count += 1
                sol1 = sols[j].tolist()
                sol2 = lkh_routes[j].tolist()

                sim = tsp_similarity(sol1, sol2)

                hamming_sum += sim["hamming_distance"]
                levenshtein_sum += sim["edit_distance"]
                lcs_sum += sim["lcs_length"]
                edge_sum += sim["edge_overlap_ratio"]
                kendall_sum += sim["kendall_tau_distance"]

        if valid_count > 0:
            print(f"Feasible pairs: {valid_count}")
            print(f"Avg Hamming distance:        {hamming_sum / valid_count:.4f}")
            print(f"Avg edit distance:           {levenshtein_sum / valid_count:.4f}")
            print(f"Avg LCS length:              {lcs_sum / valid_count:.4f}")
            print(f"Avg edge overlap ratio:      {edge_sum / valid_count:.4f}")
            print(f"Avg Kendall tau distance:    {kendall_sum / valid_count:.4f}")
        else:
            print("No feasible pairs with LKH for similarity computation.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analysis and visualization CLI refactored from the original similarity.py main block."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="tw50_traj",
        choices=["tw100_penalty", "tw50_traj", "tw50_grid", "tw50_similarity"],
        help="Which analysis routine to run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for heavy tensor computations (used in tw100_penalty).",
    )
    parser.add_argument(
        "--instances",
        type=str,
        default="10,49,20,80",
        help="Comma-separated instance indices for trajectory plotting (tw50_traj mode).",
    )
    parser.add_argument(
        "--grid_instance",
        type=int,
        default=0,
        help="Instance index used in 4x5 grid visualization (tw50_grid mode).",
    )
    parser.add_argument(
        "--grid_num",
        type=int,
        default=20,
        help="Number of solutions to visualize in the grid (tw50_grid mode).",
    )

    args = parser.parse_args()

    if args.mode == "tw100_penalty":
        analyze_tw100_car_pomo(device=args.device)
    elif args.mode == "tw50_traj":
        inst_list = [int(x) for x in args.instances.split(",") if x.strip()]
        plot_tw50_trajectories(selected_instances=inst_list)
    elif args.mode == "tw50_grid":
        visualize_tw50_solutions_grid(instance_idx=args.grid_instance, num_solutions=args.grid_num)
    elif args.mode == "tw50_similarity":
        compute_tw50_similarity_stats()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")