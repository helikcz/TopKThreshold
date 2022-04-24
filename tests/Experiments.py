import matplotlib.pyplot as plt
from collections import defaultdict

from src.Algorithms import *
from RandomGen import *


def run_queries(data, req, stats):
    for aggr_func in AGGR_FUNCTIONS:
        req.aggr_func_str = aggr_func
        _, seq_acc_seq, rand_acc_seq, time_seq = top_k_sequential(data, req)
        _, seq_acc_thresh, rand_acc_thresh, time_thresh = top_k_threshold(data, req)
        stats[(aggr_func, SEQUENTIAL)][0].append(seq_acc_seq + rand_acc_seq)
        stats[(aggr_func, SEQUENTIAL)][1].append(time_seq)

        stats[(aggr_func, THRESHOLD)][0].append(seq_acc_thresh + rand_acc_thresh)
        stats[(aggr_func, THRESHOLD)][1].append(time_thresh)


def compare_attr_count(data: WHappinessDataSrc, aggr_func_str: str):
    attr_sizes = [x for x in range(len(VAL_COLS))]
    req = RequestData()
    req.k_value = 20
    req.aggr_func_str = aggr_func_str
    stats = defaultdict(lambda: ([], []))

    for attr in VAL_COLS:
        req.attr_filter.add(attr)
        _, seq_acc_seq, rand_acc_seq, time_seq = top_k_sequential(data, req)
        _, seq_acc_thresh, rand_acc_thresh, time_thresh = top_k_threshold(data, req)

        stats[(req.aggr_func_str, SEQUENTIAL)][0].append(seq_acc_seq + rand_acc_seq)
        stats[(req.aggr_func_str, SEQUENTIAL)][1].append(time_seq)

        stats[(req.aggr_func_str, THRESHOLD)][0].append(seq_acc_thresh + rand_acc_thresh)
        stats[(req.aggr_func_str, THRESHOLD)][1].append(time_thresh)

    # plotting time graph
    plt.figure(figsize=(12, 4))
    plt.title(f"Comparison of seq/thresh top(k) with different attribute count\n (K value={req.k_value}, dataset size={data.get_size()})")
    plt.xlabel('Attr count')
    plt.ylabel('Time (s)')
    for method in METHODS:
        plt.plot(attr_sizes, stats[(req.aggr_func_str, method)][1], label=f"{method} with {req.aggr_func_str} aggr func")

    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
    plt.tight_layout()
    plt.show()

    # plotting data access
    plt.figure(figsize=(12, 4))
    plt.title(f"Comparison of seq/thresh top(k) with different attribute count\n (K value={req.k_value}, dataset size={data.get_size()})")
    plt.xlabel('Attr count')
    plt.ylabel('Data access')
    for method in METHODS:
        plt.plot(attr_sizes, stats[(req.aggr_func_str, method)][0], label=f"{method} with {req.aggr_func_str} aggr func")

    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
    plt.tight_layout()
    plt.show()


def compare_diff_k_size(data: WHappinessDataSrc):
    sizes = [10, *range(20, 101, 20)]
    req = RequestData()
    req.attr_filter = {"Social_support", "GDP_per_capita", "Score"}

    stats = defaultdict(lambda: ([], []))
    for size in sizes:
        req.k_value = size

        run_queries(data, req, stats)

    # plotting time graph
    plt.figure(figsize=(12, 4))
    plt.title(
        f"Comparison of seq/thresh top(k) with different k sizes\n ({len(req.attr_filter)} attr, dataset size={data.get_size()})")
    plt.xlabel('K size')
    plt.ylabel('Time (s)')
    for method in METHODS:
        for aggr_func in AGGR_FUNCTIONS:
            plt.plot(sizes, stats[(aggr_func, method)][1], label=f"{method} with {aggr_func} aggr func")
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
    plt.tight_layout()
    plt.show()

    # plotting data access
    plt.figure(figsize=(12, 4))
    plt.title(
        f"Comparison of seq/thresh top(k) with different k sizes\n ({len(req.attr_filter)} attr, dataset size={data.get_size()})")
    plt.xlabel('K size')
    plt.ylabel('Total data access')
    for method in METHODS:
        for aggr_func in AGGR_FUNCTIONS:
            plt.plot(sizes, stats[(aggr_func, method)][0], label=f"{method} with {aggr_func} aggr func")
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
    plt.tight_layout()
    plt.show()


def compare_diff_db_size():
    sizes = [1000, *range(2000, 20001, 2000)]
    req = RequestData()
    req.k_value = 20
    req.attr_filter = {"Social_support", "GDP_per_capita", "Score"}

    stats = defaultdict(lambda: ([], []))
    for size in sizes:
        generate_dataset(size)
        data = WHappinessDataSrc(f"../datasets/random-{size}.csv")
        run_queries(data, req, stats)

    # plotting time graph
    plt.figure(figsize=(12, 4))
    plt.title(
        f"Comparison of seq/thresh top(k) with different database sizes\n ({len(req.attr_filter)} attr, k value={req.k_value})")
    plt.xlabel('DB size')
    plt.ylabel('Time (s)')
    for method in METHODS:
        for aggr_func in AGGR_FUNCTIONS:
            plt.plot(sizes, stats[(aggr_func, method)][1], label=f"{method} with {aggr_func} aggr func")
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
    plt.tight_layout()
    plt.show()

    # plotting data access
    plt.figure(figsize=(12, 4))
    plt.title(
        f"Comparison of seq/thresh top(k) with different database sizes\n ({len(req.attr_filter)} attr, k value={req.k_value})")
    plt.xlabel('DB size')
    plt.ylabel('Total data access')
    for method in METHODS:
        for aggr_func in AGGR_FUNCTIONS:
            plt.plot(sizes, stats[(aggr_func, method)][0], label=f"{method} with {aggr_func} aggr func")
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
    plt.tight_layout()
    plt.show()

