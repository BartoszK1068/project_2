import numpy as np


def entropy(data, target):
    # Entropia mierzy nieuporzadkowanie klas w zbiorze.
    probabilities = data[target].value_counts(normalize=True).to_numpy()
    return float(-(probabilities * np.log2(probabilities)).sum())


def split_rows(data, attribute):
    # Podzial danych na grupy wedlug jednej wartosci atrybutu.
    groups = {}
    for value in data[attribute].unique():
        groups[value] = data[data[attribute] == value]
    return groups


def info_gain(data, attribute, target):
    # Zysk informacji porownuje entropie przed i po podziale.
    start_entropy = entropy(data, target)
    groups = split_rows(data, attribute)
    after_split = 0.0

    for group in groups.values():
        after_split += (len(group) / len(data)) * entropy(group, target)

    return start_entropy - after_split


def split_info(data, attribute):
    # SplitInfo opisuje "koszt" podzialu na wiele galezi.
    groups = split_rows(data, attribute)
    result = 0.0

    for group in groups.values():
        p = len(group) / len(data)
        result -= p * np.log2(p)

    return float(result)


def gain_ratio(data, attribute, target):
    # GainRatio normalizuje InfoGain przez SplitInfo.
    gain = info_gain(data, attribute, target)
    value = split_info(data, attribute)
    if value == 0:
        return 0
    return gain / value


def majority_class(data, target):
    return data[target].mode()[0]


def choose_best_attribute(data, attributes, target, criterion):
    best_attribute = None
    best_score = -1

    for attribute in attributes:
        if criterion == "info_gain":
            score = info_gain(data, attribute, target)
        else:
            score = gain_ratio(data, attribute, target)

        if score > best_score:
            best_score = score
            best_attribute = attribute

    return best_attribute


def build_tree(data, attributes, target, criterion):
    # Rekurencyjna budowa drzewa decyzyjnego.
    classes = data[target].tolist()

    if len(set(classes)) == 1:
        return classes[0]

    if len(attributes) == 0:
        return majority_class(data, target)

    best_attribute = choose_best_attribute(data, attributes, target, criterion)
    if best_attribute is None:
        return majority_class(data, target)

    tree = {
        "attribute": best_attribute,
        "default": majority_class(data, target),
        "children": {},
    }

    groups = split_rows(data, best_attribute)
    remaining_attributes = [a for a in attributes if a != best_attribute]

    for value, group in groups.items():
        tree["children"][value] = build_tree(group, remaining_attributes, target, criterion)

    return tree


def classify(tree, sample):
    # Klasyfikacja jednej nowej probki na podstawie gotowego drzewa.
    if not isinstance(tree, dict):
        return tree

    attribute = tree["attribute"]
    value = sample.get(attribute)

    if value not in tree["children"]:
        return tree["default"]

    return classify(tree["children"][value], sample)


def count_nodes(tree):
    if not isinstance(tree, dict):
        return 1

    total = 1
    for child in tree["children"].values():
        total += count_nodes(child)
    return total


def tree_depth(tree):
    if not isinstance(tree, dict):
        return 1

    depths = []
    for child in tree["children"].values():
        depths.append(tree_depth(child))
    return 1 + max(depths)


def root_attribute(tree):
    if not isinstance(tree, dict):
        return "leaf"
    return tree["attribute"]


def tree_to_text(tree, level=0):
    # Zamiana drzewa na tekst, aby latwo zapisac je do pliku.
    indent = "  " * level

    if not isinstance(tree, dict):
        return indent + "Klasa: " + str(tree)

    lines = [indent + "Atrybut: " + tree["attribute"]]
    for value, child in tree["children"].items():
        lines.append(indent + "- wartosc = " + str(value))
        lines.append(tree_to_text(child, level + 1))
    return "\n".join(lines)


def print_tree(tree, level=0):
    print(tree_to_text(tree, level))
