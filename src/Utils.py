import time
from dataclasses import *
from flask import request

DEFAULT_K = 10
SEQUENTIAL, THRESHOLD = "sequential", "threshold"
METHODS = [SEQUENTIAL, THRESHOLD]

AGGR_MAX, AGGR_AVG, AGGR_MIN = "max", "avg", "min"
AGGR_FUNCTIONS = [AGGR_MAX, AGGR_AVG, AGGR_MIN]

VALID_FILTER_VAL = {"Score", "GDP_per_capita", "Social_support", "Healthy_life_expectancy",
                    "Freedom_to_make_life_choices", "Generosity", "Perceptions_of_corruption"}


def timer_decorator(func):
    def wrap_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return (*result, end - start)
    return wrap_func


class RequestParser:
    @staticmethod
    def parse(req: request):
        aggr_func = req.args.get('aggr')
        aggr_func = aggr_func if aggr_func in AGGR_FUNCTIONS else AGGR_MAX

        req_sort_filter = req.args.getlist('sort_by')
        attr_filter = VALID_FILTER_VAL.intersection(req_sort_filter)

        if len(attr_filter) == 0:
            attr_filter.add("Score")

        k_value = req.args.get("k_value", type=int)
        k_value = int(DEFAULT_K if k_value is None else k_value)

        method = req.args.get('queryMethod')
        method = method if method in METHODS else THRESHOLD

        return RequestData(aggr_func, attr_filter, k_value, method)


@dataclass(init=True)
class RequestData:
    aggr_func_str: str = field(default=AGGR_MAX)
    attr_filter: set = field(default_factory=lambda: set())
    k_value: int = field(default=DEFAULT_K)
    method: str = field(default=SEQUENTIAL)
