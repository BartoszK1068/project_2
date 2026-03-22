import pandas as pd


DEFAULT_DATA_PATH = "dane/road_signs.csv"
TARGET_COLUMN = "sign_class"
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


def split_train_test(data, target):
    # Prosty podzial warstwowy: czesc do treningu i czesc do testu.
    train_parts = []
    test_parts = []

    for _, group in data.groupby(target):
        split_index = max(1, len(group) - max(1, len(group) // 3))
        train_parts.append(group.iloc[:split_index])
        test_parts.append(group.iloc[split_index:])

    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    return train, test
