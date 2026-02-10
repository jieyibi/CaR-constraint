import torch
import numpy as np


def hamming_distance_vectorized(solutions):
    """Compute Hamming distance in a vectorized manner"""
    # solutions: (solution_size, solution_length)
    solution_size, solution_length = solutions.shape

    # Expand dimensions for broadcasting comparison
    sol_expanded1 = solutions.unsqueeze(1)  # (solution_size, 1, solution_length)
    sol_expanded2 = solutions.unsqueeze(0)  # (1, solution_size, solution_length)

    # Compute Hamming distances between all pairs
    distances = (sol_expanded1 != sol_expanded2).sum(dim=2).float()  # (solution_size, solution_size)

    # Only take upper triangular part (avoid duplicate computation)
    triu_indices = torch.triu_indices(solution_size, solution_size, offset=1, device=solutions.device)
    return distances[triu_indices[0], triu_indices[1]]


def jaccard_distance_vectorized(solutions):
    """Compute Jaccard distance in a vectorized manner (based on set similarity)"""
    solution_size, solution_length = solutions.shape
    device = solutions.device

    # Create one-hot encoding for each solution
    max_val = solutions.max().item() + 1
    one_hot = torch.zeros(solution_size, max_val, device=device)
    one_hot.scatter_(1, solutions, 1)

    # Compute intersection and union
    intersection = torch.matmul(one_hot, one_hot.t())  # (solution_size, solution_size)
    union = one_hot.sum(dim=1, keepdim=True) + one_hot.sum(dim=1, keepdim=True).t() - intersection

    # Compute Jaccard distance
    jaccard_sim = intersection / (union + 1e-8)
    jaccard_dist = 1 - jaccard_sim

    # Only take upper triangular part
    triu_indices = torch.triu_indices(solution_size, solution_size, offset=1, device=device)
    return jaccard_dist[triu_indices[0], triu_indices[1]]


def positional_jaccard_distance(solutions):
    """Position-based Jaccard distance - more suitable for permutation problems"""
    solution_size, solution_length = solutions.shape
    device = solutions.device

    distances = []
    for i in range(solution_size):
        for j in range(i + 1, solution_size):
            # Count the number of same elements at the same positions
            same_positions = (solutions[i] == solutions[j]).sum().item()
            # Jaccard similarity based on position matching
            similarity = same_positions / solution_length
            distance = 1 - similarity
            distances.append(distance)

    return torch.tensor(distances, device=device)


def edge_set_jaccard_distance(solutions: torch.Tensor) -> torch.Tensor:
    """
    Jaccard distance based on undirected edge sets (same input/output format as positional_jaccard_distance)

    solutions: [solution_size, solution_length]
        Each row is a CVRP solution (contains depot 0, may have padding 0 at the end)

    Returns:
        distances: [num_pairs], ordered as (0,1), (0,2), ..., (0,n-1), (1,2), ...
    """
    device = solutions.device
    solution_size, solution_length = solutions.shape

    # Adjacent points form edges
    u = solutions[:, :-1]  # [S, L-1]
    v = solutions[:,  1:]  # [S, L-1]

    # Undirected edges: a <= b
    a = torch.minimum(u, v)
    b = torch.maximum(u, v)

    # Skip padding segments 0->0 and self-loops
    mask = ~((a == 0) & (b == 0)) & (a != b)

    # Encode edge ID with (a, b), using full grid encoding: edge_id ∈ [0, num_nodes^2)
    max_node = solutions.max()
    num_nodes = int(max_node.item()) + 1
    total_edges = num_nodes * num_nodes

    edge_id = a * num_nodes + b  # [S, L-1]

    # Construct edge indicator matrix X: [S, total_edges], each row indicates which undirected edges the solution contains
    X = torch.zeros((solution_size, total_edges),
                    dtype=torch.float32, device=device)

    rows = torch.arange(solution_size, device=device).unsqueeze(1).expand_as(edge_id)
    X[rows[mask], edge_id[mask]] = 1.0

    # Number of edges for each solution |E_i|
    card = X.sum(dim=1)  # [S]

    # Intersection size |E_i ∩ E_j|: binary vector dot product
    inter = X @ X.T      # [S, S]

    # Union size |E_i ∪ E_j| = |E_i| + |E_j| - |E_i ∩ E_j|
    union = card.unsqueeze(0) + card.unsqueeze(1) - inter  # [S, S]

    # Jaccard distance
    dist = torch.zeros_like(inter)
    valid = union > 0
    dist[valid] = 1.0 - inter[valid] / union[valid]

    # Same as positional_jaccard_distance: return upper triangular (i<j) flattened
    idx_i, idx_j = torch.triu_indices(solution_size, solution_size, offset=1)
    distances = dist[idx_i, idx_j]

    return distances


def kendall_tau_distance_vectorized(solutions):
    """Compute Kendall's tau distance in a vectorized manner"""
    solution_size, solution_length = solutions.shape
    device = solutions.device

    # Convert solutions to rankings
    # For each solution, create a mapping from values to positions
    ranks = torch.zeros_like(solutions)
    for i in range(solution_size):
        # Create ranking mapping: position of each value in the sequence
        unique_vals = solutions[i].unique()
        for j, val in enumerate(unique_vals):
            mask = solutions[i] == val
            ranks[i][mask] = torch.where(solutions[i] == val)[0][0]

    # Compute Kendall tau distance between all solution pairs
    distances = []
    solutions_np = solutions.cpu().numpy()

    for i in range(solution_size):
        for j in range(i + 1, solution_size):
            seq1, seq2 = solutions_np[i], solutions_np[j]

            # Count the number of discordant pairs
            discordant_pairs = 0
            total_pairs = 0

            for p in range(solution_length):
                for q in range(p + 1, solution_length):
                    val1_p, val1_q = seq1[p], seq1[q]
                    val2_p, val2_q = seq2[p], seq2[q]

                    # Check if the relative order of these two values is consistent in both sequences
                    pos1_in_seq2_p = np.where(seq2 == val1_p)[0]
                    pos1_in_seq2_q = np.where(seq2 == val1_q)[0]

                    if len(pos1_in_seq2_p) > 0 and len(pos1_in_seq2_q) > 0:
                        total_pairs += 1
                        # Check if the order is inconsistent
                        if (p < q) != (pos1_in_seq2_p[0] < pos1_in_seq2_q[0]):
                            discordant_pairs += 1

            # Kendall tau distance
            tau_distance = discordant_pairs / total_pairs if total_pairs > 0 else 0
            distances.append(tau_distance)

    return torch.tensor(distances, device=device)


def calculate_diversity_gpu_optimized(solutions_tensor, distance_metric='hamming'):
    """
    GPU-optimized version: compute diversity for all instances at once

    Args:
        solutions_tensor: tensor of shape (solution_size, instance_size, solution_length)
        distance_metric: 'hamming', 'jaccard', 'positional_jaccard', or 'kendall'

    Returns:
        overall_diversity: average diversity
        instance_diversities: list of diversity for each instance
    """
    solution_size, instance_size, solution_length = solutions_tensor.shape
    device = solutions_tensor.device

    print(f"Computing diversity for {instance_size} instances, each with {solution_size} solutions...")
    print(f"Using distance metric: {distance_metric}")

    if distance_metric == 'hamming':
        # Reshape tensor: (instance_size, solution_size, solution_length)
        solutions_reordered = solutions_tensor.permute(1, 0, 2)

        # Expand dimensions for broadcasting
        sol_exp1 = solutions_reordered.unsqueeze(2)  # (instance_size, solution_size, 1, solution_length)
        sol_exp2 = solutions_reordered.unsqueeze(1)  # (instance_size, 1, solution_size, solution_length)

        # Compute Hamming distances between all solution pairs for all instances
        distances = (sol_exp1 != sol_exp2).sum(dim=3).float()  # (instance_size, solution_size, solution_size)

        # Only take upper triangular part
        triu_mask = torch.triu(torch.ones(solution_size, solution_size, device=device), diagonal=1).bool()

        # Compute average distance for each instance
        instance_diversities = []
        for i in range(instance_size):
            upper_tri_distances = distances[i][triu_mask]
            avg_diversity = upper_tri_distances.mean().item()
            instance_diversities.append(avg_diversity)

    elif distance_metric in ['jaccard', 'positional_jaccard', 'kendall', "edge_jaccard"]:
        # Batch processing to save memory and computation time
        batch_size = min(50, instance_size)  # Reduce batch size to accommodate complex computations
        instance_diversities = []

        # Select distance function
        if distance_metric == 'jaccard':
            distance_func = jaccard_distance_vectorized
        elif distance_metric == 'edge_jaccard':
            distance_func = edge_set_jaccard_distance
        elif distance_metric == 'positional_jaccard':
            distance_func = positional_jaccard_distance
        else:  # kendall
            distance_func = kendall_tau_distance_vectorized

        for batch_start in range(0, instance_size, batch_size):
            batch_end = min(batch_start + batch_size, instance_size)
            batch_diversities = []

            for instance_idx in range(batch_start, batch_end):
                current_solutions = solutions_tensor[:, instance_idx, :]
                try:
                    distances = distance_func(current_solutions)
                    avg_diversity = distances.mean().item() if len(distances) > 0 else 0.0
                    batch_diversities.append(avg_diversity)
                except Exception as e:
                    print(f"Error computing instance {instance_idx}: {e}")
                    batch_diversities.append(0.0)

            instance_diversities.extend(batch_diversities)

            if (batch_start // batch_size + 1) % 10 == 0:
                print(f"Processed: {batch_end}/{instance_size} instances")

    else:
        raise ValueError(
            f"Unsupported distance metric: {distance_metric}, supported options: 'hamming', 'jaccard', 'positional_jaccard', 'kendall'")

    overall_diversity = np.mean(instance_diversities)
    return overall_diversity, instance_diversities


# Main computation function
def compute_solution_diversity(file_path=None, distance_metrics=['hamming', 'positional_jaccard', 'kendall'], solutions=None):
    """
    Main function to compute solution diversity

    Args:
        file_path: path to .pt file
        distance_metrics: list of distance metric methods
    """
    print(f"Loading solution data: {file_path}")
    if solutions is None:
        solutions = torch.load(file_path)

    print(f"Data shape: {solutions.shape}")
    print(f"Data type: {solutions.dtype}")

    # Check data range
    print(f"Data range: [{solutions.min().item()}, {solutions.max().item()}]")

    # Use GPU acceleration if available
    if torch.cuda.is_available():
        solutions = solutions.cuda()
        print("Using GPU computation")
    else:
        print("Using CPU computation")

    results = {}

    # Compute diversity for different distance metrics
    for metric in distance_metrics:
        print(f"\n--- Computing {metric} distance ---")
        try:
            overall_diversity, instance_diversities = calculate_diversity_gpu_optimized(solutions, metric)

            results[metric] = {
                'overall': overall_diversity,
                'instances': instance_diversities,
                'std': np.std(instance_diversities),
                'min': np.min(instance_diversities),
                'max': np.max(instance_diversities)
            }

            print(f"Average diversity: {overall_diversity:.6f}")
            print(f"Diversity std: {np.std(instance_diversities):.6f}")
            print(f"Diversity range: [{np.min(instance_diversities):.6f}, {np.max(instance_diversities):.6f}]")

        except Exception as e:
            print(f"Error computing {metric} distance: {e}")
            results[metric] = None

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default="./tw50_car_init_solution_sample.pt")
    parser.add_argument('--file2', type=str, default="./tw50_pip_solution.pt")
    parser.add_argument('--metrics', type=str, default="hamming,positional_jaccard,kendall", help="Comma-separated list of distance metrics to compute")
    args = parser.parse_args()
    metrics = args.metrics.split(',')
    results1 = compute_solution_diversity(args.file1, args.metrics)
    results2 = compute_solution_diversity(args.file2, args.metrics)

    # Compare results
    print("\n" + "=" * 80)
    print("Diversity Comparison Summary")
    print("=" * 80)

    for metric in metrics:
        if metric in results1 and metric in results2 and results1[metric] and results2[metric]:
            print(f"\n{metric.upper()} distance:")
            print(f"  File1 (car_init): {results1[metric]['overall']:.6f} (±{results1[metric]['std']:.6f})")
            print(f"  File2 (pip):      {results2[metric]['overall']:.6f} (±{results2[metric]['std']:.6f})")
            if results1[metric]['overall'] > 0:
                improvement = (results2[metric]['overall'] - results1[metric]['overall']) / results1[metric][
                    'overall'] * 100
                print(f"  Improvement: {improvement:+.2f}%")
        else:
            print(f"\n{metric.upper()} distance: Computation failed or data incomplete")

    print("\n" + "=" * 80)
    print("Notes:")
    print("- Hamming distance: number of different elements at the same positions")
    print("- Positional Jaccard distance: similarity based on position matching")
    print("- Kendall distance: inconsistency based on relative order of elements")
    print("- Higher values indicate greater diversity")
    print("=" * 80)