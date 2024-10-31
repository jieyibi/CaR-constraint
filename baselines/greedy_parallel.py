import pickle
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_data(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def save_results(results, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def greedy_tsptw_instance(instance, heuristics):
    node_xy, service_time, tw_start, tw_end = instance
    num_nodes = len(node_xy)
    visited = [False] * num_nodes
    visited[0] = True  # Start at node 0
    current_node = 0
    tour = [current_node]
    total_distance = 0
    current_time = 0
    feasible = True

    while len(tour) < num_nodes:
        next_node = None
        min_tw_end = float('inf')
        min_distance = float('inf')
        for i in range(1, num_nodes):  # Start from 1 as 0 is the depot
            if not visited[i]:
                if heuristics == "twend":
                    if tw_end[i] < min_tw_end:
                        min_tw_end = tw_end[i]
                        next_node = i
                elif heuristics == "length":
                    distance = calculate_distance(node_xy[current_node], node_xy[i])
                    if distance < min_distance:
                        min_distance = distance
                        next_node = i
                else:
                    raise NotImplementedError
        if next_node is None:
            break
        visited[next_node] = True
        tour.append(next_node)
        travel_time = calculate_distance(node_xy[current_node], node_xy[next_node]) if heuristics != "length" else min_distance
        current_time = max(current_time + travel_time, tw_start[next_node])
        if current_time > tw_end[next_node] + 0.000001:
            feasible = False
        total_distance += travel_time
        current_node = next_node

    # Return to start node to complete the tour
    total_distance += calculate_distance(node_xy[current_node], node_xy[0])
    tour.append(0)  # Append start node to complete the cycle

    return (tour, total_distance, feasible)

def greedy_tsptw_algorithm(instances, heuristics, num_workers=16):
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_instance = {executor.submit(greedy_tsptw_instance, instance, heuristics): instance for instance in instances}
        for future in as_completed(future_to_instance):
            result = future.result()
            results.append(result)
    return results

def greedy_tspdl_instance(instance, heuristics):
    node_xy, demand, draft_limit = instance
    num_nodes = len(node_xy)
    visited = [False] * num_nodes
    visited[0] = True  # Start at node 0
    current_node = 0
    tour = [current_node]
    total_distance = 0
    current_load = 0
    feasible = True

    while len(tour) < num_nodes:
        next_node = None
        min_draft_limit = float('inf')
        min_distance = float('inf')
        for i in range(1, num_nodes):  # Start from 1 as 0 is the depot
            if not visited[i]:
                if heuristics == "dl":
                    if draft_limit[i] < min_draft_limit:
                        min_draft_limit = draft_limit[i]
                        next_node = i
                elif heuristics == "length":
                    distance = calculate_distance(node_xy[current_node], node_xy[i])
                    if distance < min_distance:
                        min_distance = distance
                        next_node = i
                else:
                    raise NotImplementedError
        if next_node is None:
            break
        visited[next_node] = True
        tour.append(next_node)
        travel_time = calculate_distance(node_xy[current_node], node_xy[next_node]) if heuristics != "length" else min_distance
        current_load += demand[next_node]
        if current_load > draft_limit[next_node] + 0.000001:
            feasible = False
        total_distance += travel_time
        current_node = next_node

    # Return to start node to complete the tour
    total_distance += calculate_distance(node_xy[current_node], node_xy[0])
    tour.append(0)  # Append start node to complete the cycle

    return (tour, total_distance, feasible)

def greedy_tspdl_algorithm(instances, heuristics, num_workers=16):
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_instance = {executor.submit(greedy_tspdl_instance, instance, heuristics): instance for instance in instances}
        for future in as_completed(future_to_instance):
            result = future.result()
            results.append(result)
    return results

if __name__ == "__main__":
    # Example usage
    # heuristics = "dl"
    # heuristics = "twend"
    heuristics = "length"
    # filepath = '/home/jieyi/Routing-Anything-main/data/TSPTW/tsptw100_da_silva_uniform.pkl'
    # save_path = '/home/jieyi/Routing-Anything-main/data/TSPTW/greedy_{}_tsptw100_da_silva_uniform.pkl'.format(heuristics)
    # lkh = '/home/jieyi/Routing-Anything-main/data/TSPTW/lkh_tsptw100_da_silva_uniform.pkl'
    filepath = '/home/jieyi/Routing-Anything-main/data/TSPTW/tsptw50_zhang_uniform_1020.pkl'
    save_path = '/home/jieyi/Routing-Anything-main/data/TSPTW/greedy_{}_tsptw50_zhang_uniform_1020.pkl'.format(heuristics)
    lkh = '/home/jieyi/Routing-Anything-main/data/TSPTW/lkh_tsptw50_zhang_uniform_1020.pkl'
    # filepath = "/home/jieyi/Routing-Anything-main/data/TSPDL/tspdl50_q90_uniform_fsb.pkl"
    # lkh = "/home/jieyi/Routing-Anything-main/data/TSPDL/lkh_tspdl50_q90_uniform_fsb.pkl"
    # save_path = '/home/jieyi/Routing-Anything-main/data/TSPDL/greedy_{}_tspdl50_q90_uniform_fsb.pkl'.format(heuristics)
    start = time.time()
    data = load_data(filepath)

    # results = greedy_tspdl_algorithm(data, heuristics)
    results = greedy_tsptw_algorithm(data, heuristics)

    duration = time.time() - start

    save_results(results, save_path)
    with open(lkh, 'rb') as file:
        lkh = pickle.load(file)

    gaps = np.array([])
    distances = np.array([])
    opt = np.array([])
    feasible_cnt = 0
    # Optionally, print or process results further
    for i, result in enumerate(results):
        opt = np.append(opt, lkh[i][0])
        gap = (result[1] - lkh[i][0]) / lkh[i][0] *100
        print("Tour {}:".format(i), result[0], "Total Distance:", result[1], "Optimal:",lkh[i][0],  "Gap:{}%".format(gap), "Feasible:", result[2])
        if result[2]:
            distances = np.append(distances, result[1] / 100)
            gaps = np.append(gaps, gap)
            feasible_cnt += 1

    print("Duration: {}".format(duration))
    print("Average distance: {}".format(np.mean(distances)))
    print("Average distance in LKH3: {}".format(np.mean(opt)))
    print("Average gap: {}%".format(np.mean(gaps)))
    print("Infeasible count: {}".format(len(results)-feasible_cnt))
