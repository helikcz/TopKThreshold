import heapq
from typing import List

from src.DataSource import *
from src.Utils import *


def get_aggr_func(name: str):
    # returns aggregation function for given name
    if name == AGGR_MAX:
        return lambda x: max(x)
    elif name == AGGR_AVG:
        return lambda x: round(sum(x) / len(x), 4)
    else:
        return lambda x: min(x)


@timer_decorator
def top_k_sequential(data: WHappinessDataSrc, requestData: RequestData) -> [List[WHappiness], int]:
    seq_acc, random_acc = 0, 0
    if requestData.k_value <= 0:
        return [], seq_acc, random_acc
    aggr_func = get_aggr_func(requestData.aggr_func_str)

    result: List[WHappiness] = []
    for state in data.data.itertuples():
        seq_acc += 1
        state_normalized = data.data_normalized.loc[state.Index]

        val = aggr_func([state_normalized[attribute]
                         for attribute in requestData.attr_filter])

        line = WHappiness(state.Index, state.Country_or_region, state.Score,
                          state.GDP_per_capita, state.Social_support, state.Healthy_life_expectancy,
                          state.Freedom_to_make_life_choices, state.Generosity,
                          state.Perceptions_of_corruption, val)
        result.append(line)

    result = sorted(result, key=lambda x: (x.Aggregate, -x.Overall_rank), reverse=True)
    return result[:requestData.k_value], seq_acc, random_acc


@timer_decorator
def top_k_threshold(data: WHappinessDataSrc, requestData: RequestData) -> [List[WHappiness], int]:
    seq_acc, random_acc = 0, 0
    if requestData.k_value <= 0:
        return [], seq_acc, random_acc
    aggr_func = get_aggr_func(requestData.aggr_func_str)

    # python minHeap
    heap = []
    heapq.heapify(heap)

    seen_states = set()

    # iteration over ordered lists
    for i in range(data.get_size()):
        seq_acc += 1

        threshold = aggr_func([data.sorted_cols_data[attr][i][1]
                               for attr in requestData.attr_filter])

        # reached threshold (and k amount), no better item is possible
        if len(heap) >= requestData.k_value and (heap[0][0]) >= threshold:
            break

        for attr in requestData.attr_filter:
            index, val = data.sorted_cols_data[attr][i]

            state = data.data.loc[index]
            if index not in seen_states:
                random_acc += 1
                seen_states.add(index)

                state_normalized = data.data_normalized.loc[index]

                val = aggr_func([state_normalized[attribute]
                                 for attribute in requestData.attr_filter])

                line = WHappiness(index, state.Country_or_region, state.Score,
                                  state.GDP_per_capita, state.Social_support, state.Healthy_life_expectancy,
                                  state.Freedom_to_make_life_choices, state.Generosity,
                                  state.Perceptions_of_corruption, val)

                heapq.heappush(heap, (val, line))

                if len(heap) > requestData.k_value:
                    heapq.heappop(heap)

    result = [state[1] for state in heapq.nlargest(requestData.k_value, heap)]

    # sort to make it more consistent with naive (maybe not necessary)
    # by aggregate already sorted, only flips if Aggregate are same and wrong overall order
    result = sorted(result, key=lambda x: (x.Aggregate, -x.Overall_rank), reverse=True)
    return result, seq_acc, random_acc
