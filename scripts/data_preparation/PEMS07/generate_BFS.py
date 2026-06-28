import os
import sys
import shutil
import pickle
import argparse

import numpy as np

from generate_adj_mx import generate_adj_pems08
# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from basicts.data.transform import standard_transform


import collections


def get_shortest_path_distance_matrix(connectivity_mx: np.array) -> np.array:
    N = connectivity_mx.shape[0]
    
    dist_mx = np.full((N, N), np.inf)
    np.fill_diagonal(dist_mx, 0)

   
    graph = collections.defaultdict(list)
    for i in range(N):
        for j in range(N):
            if connectivity_mx[i, j] == 1:
                graph[i].append(j)

    
    print("Calculating all-pairs shortest paths...")
    for start_node in range(N):
        queue = collections.deque([(start_node, 0)])  
        visited = {start_node}

        while queue:
            current_node, distance = queue.popleft()

            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist_mx[start_node, neighbor] = distance + 1
                    queue.append((neighbor, distance + 1))

    print("Shortest path calculation complete.")
    return dist_mx


def generate_weighted_adj(distance_mx: np.array, connectivity_mx: np.array, sigma2=10, epsilon=0.5) -> np.array:
    
    print("Calculating weights based on the provided distance matrix...")
    weights = np.exp(-np.square(distance_mx / np.sqrt(sigma2)))
    weights[weights < epsilon] = 0
    final_weights = weights * connectivity_mx
    np.fill_diagonal(final_weights, 1)
    return final_weights

def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.

    Args:
        args (argparse): configurations of preprocessing
    """

    target_channel = args.target_channel
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    graph_file_path = args.graph_file_path
    steps_per_day = args.steps_per_day
    norm_each_channel = args.norm_each_channel
    if_rescale = not norm_each_channel

    
    data = np.load(data_file_path)["data"]
    data = data[..., target_channel]
    print("raw time series shape: {0}".format(data.shape))
    l, n, f = data.shape
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num = round(num_samples * train_ratio)
    valid_num = round(num_samples * valid_ratio)
    test_num = num_samples - train_num - valid_num
    print("number of training samples:{0}".format(train_num))
    print("number of validation samples:{0}".format(valid_num))
    print("number of test samples:{0}".format(test_num))

    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t-history_seq_len, t, t+future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    valid_index = index_list[train_num: train_num + valid_num]
    test_index = index_list[train_num +
                            valid_num: train_num + valid_num + test_num]

    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len, norm_each_channel=norm_each_channel)

    feature_list = [data_norm]
    if add_time_of_day:
        tod = [i % steps_per_day /
               steps_per_day for i in range(data_norm.shape[0])]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        dow = [(i // steps_per_day) % 7 / 7 for i in range(data_norm.shape[0])]
        dow = np.array(dow)
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    processed_data = np.concatenate(feature_list, axis=-1)

    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(index, f)

    data = {}
    data["processed_data"] = processed_data
    with open(output_dir + "/data_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(data, f)

    print("Generating adjacency matrix based on shortest path...")

    unweighted_adj_path = args.graph_file_path
    try:
        with open(unweighted_adj_path, 'rb') as f:
            connectivity_mx = pickle.load(f)

        assert connectivity_mx.ndim == 2, "Connectivity matrix must be 2D."
        print(f"Loaded connectivity matrix with shape: {connectivity_mx.shape}")

    except Exception as e:
        print(f"Error loading connectivity graph file: {e}")
        return

    
    shortest_path_dist_mx = get_shortest_path_distance_matrix(connectivity_mx)
    final_weighted_adj = generate_weighted_adj(shortest_path_dist_mx, np.ones_like(connectivity_mx))

    if final_weighted_adj is not None:
        final_adj_path = os.path.join(args.output_dir, "adj_mx.pkl")
        with open(final_adj_path, "wb") as f:
            pickle.dump(final_weighted_adj, f)
        print(f"Final shortest-path based weighted adjacency matrix saved to {final_adj_path}")

if __name__ == "__main__":
    HISTORY_SEQ_LEN = 12
    FUTURE_SEQ_LEN = 12
    TRAIN_RATIO = 0.6
    VALID_RATIO = 0.2
    TARGET_CHANNEL = [0]
    STEPS_PER_DAY = 288
    DATASET_NAME = "PEMS07"
    TOD = True
    DOW = True
    OUTPUT_DIR = "datasets/" + DATASET_NAME
    DATA_FILE_PATH = "datasets/{0}/{0}.npz".format(DATASET_NAME)
    GRAPH_FILE_PATH = "datasets/{0}/adj_{0}.pkl".format(DATASET_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                            default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                            default=DATA_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--graph_file_path", type=str,
                            default=GRAPH_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--history_seq_len", type=int,
                            default=HISTORY_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--future_seq_len", type=int,
                            default=FUTURE_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--steps_per_day", type=int,
                            default=STEPS_PER_DAY, help="Sequence Length.")
    parser.add_argument("--tod", type=bool, default=TOD,
                            help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW,
                            help="Add feature day_of_week.")
    parser.add_argument("--target_channel", type=list,
                            default=TARGET_CHANNEL, help="Selected channels.")
    parser.add_argument("--train_ratio", type=float,
                            default=TRAIN_RATIO, help="Train ratio")
    parser.add_argument("--valid_ratio", type=float,
                            default=VALID_RATIO, help="Validate ratio.")
    parser.add_argument("--norm_each_channel", type=float, help="Validate ratio.")

    args = parser.parse_args()

    print("-" * (20 + 45 + 5))
    for key, value in sorted(vars(args).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-" * (20 + 45 + 5))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.norm_each_channel = True
    generate_data(args)
    args.norm_each_channel = False
    generate_data(args)