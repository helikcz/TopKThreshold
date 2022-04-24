from dataclasses import *
import pandas as pd


VAL_COLS = ["Score", "GDP_per_capita", "Social_support", "Healthy_life_expectancy",
            "Freedom_to_make_life_choices", "Generosity", "Perceptions_of_corruption"]
COLS = ["Overall_rank", "Country_or_region", *VAL_COLS[:], "Aggregate"]


class WHappinessDataSrc:
    """
    Loads db from csv file.
    Normalizes data and presorts per column.
    """
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data.set_index("Overall_rank", inplace=True)

        self.data_normalized = WHappinessDataSrc.normalize_data(self.data)

        self.sorted_cols_data = {
            "Score": list(self.data_normalized["Score"].sort_values(ascending=False).iteritems()),
            "GDP_per_capita": list(self.data_normalized["GDP_per_capita"].sort_values(ascending=False).iteritems()),
            "Social_support": list(self.data_normalized["Social_support"].sort_values(ascending=False).iteritems()),
            "Healthy_life_expectancy": list(self.data_normalized["Healthy_life_expectancy"].sort_values(ascending=False).iteritems()),
            "Freedom_to_make_life_choices": list(self.data_normalized["Freedom_to_make_life_choices"].sort_values(ascending=False).iteritems()),
            "Generosity": list(self.data_normalized["Generosity"].sort_values(ascending=False).iteritems()),
            "Perceptions_of_corruption": list(self.data_normalized["Perceptions_of_corruption"].sort_values(ascending=False).iteritems())
        }

    def get_size(self):
        return len(self.data.index)

    @staticmethod
    def normalize_data(data):
        normalized = data.copy()
        for col in VAL_COLS:
            max_val = data[col].max()
            min_val = data[col].min()

            normalized[col] = round((data[col] - min_val) / (max_val - min_val), 4)
        return normalized


@dataclass(order=True, frozen=True)
class WHappiness:
    Overall_rank: int
    Country_or_region: str
    Score: float
    GDP_per_capita: float
    Social_support: float
    Healthy_life_expectancy: float
    Freedom_to_make_life_choices: float
    Generosity: float
    Perceptions_of_corruption: float
    Aggregate: float
