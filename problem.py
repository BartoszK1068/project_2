import pandas as pd


DEFAULT_DATA_PATH = "dane/road_signs.csv"
TARGET_COLUMN = "sign_class"
RANDOM_SEED = 42
VARIANTS = [
    "full",
    "without_crossed_out",
    "without_element_count",
    "without_shape_and_border",
]


def load_data(path):
    # Wczytanie tabeli opisujacej znaki drogowe.
    return pd.read_csv(path)


def get_attributes(data, target):
    return [column for column in data.columns if column != target]


def prepare_variant(data, target, variant_name):
    # Tworzenie wariantu z pominietymi atrybutami.
    removed = []

    if variant_name == "without_crossed_out":
        removed = ["crossed_out"]
    elif variant_name == "without_element_count":
        removed = ["element_count"]
    elif variant_name == "without_shape_and_border":
        removed = ["shape", "border_color"]

    new_data = data.drop(columns=removed)
    attributes = get_attributes(new_data, target)
    return new_data, attributes, removed


def split_train_test(data, target, train_ratio=0.7, seed=RANDOM_SEED):
    # Stratyfikowany podzial z tasowaniem w obrebie kazdej klasy.
    train_parts = []
    test_parts = []

    for _, group in data.groupby(target):
        shuffled_group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        split_index = max(1, int(round(len(shuffled_group) * train_ratio)))
        split_index = min(split_index, len(shuffled_group) - 1)
        train_parts.append(shuffled_group.iloc[:split_index])
        test_parts.append(shuffled_group.iloc[split_index:])

    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    return train, test
