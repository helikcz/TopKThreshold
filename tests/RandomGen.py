import numpy as np
import random
import string
from src.DataSource import *


def generate_dataset(size):
    df = pd.DataFrame(np.round(np.random.uniform(0, 1, size=(size, len(VAL_COLS))), 4), columns=VAL_COLS)
    df["Country_or_region"] = [
                                ''.join(
                                        random.choice(string.ascii_lowercase)
                                        for _ in range(random.randint(4, 8))
                                        )
                                for _ in range(size)
                               ]

    df.index.name = "Overall_rank"
    df.to_csv(f"../datasets/random-{size}.csv")
