import pandas as pd
import os

LOAD_PATH = "../data_set/dimension13_set"


def load_classify_data(load_path=LOAD_PATH):
    csv_path = os.path.join(load_path, "sample13_set_01.csv")
    return pd.read_csv(csv_path)


def load_test_data(load_path=LOAD_PATH):
    csv_path = os.path.join(load_path, "sample13_set_test.csv")
    return pd.read_csv(csv_path)


def load_test02_data(load_path=LOAD_PATH):
    csv_path = os.path.join(load_path, "sample13_set_test_02.csv")
    return pd.read_csv(csv_path)


def load_test03_data(load_path=LOAD_PATH):
    csv_path = os.path.join(load_path, "sample13_set_test_03.csv")
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    load_classify_data(LOAD_PATH)
