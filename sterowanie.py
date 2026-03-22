from datetime import datetime
from pathlib import Path

from metoda import (
    build_tree,
    classify,
    count_nodes,
    print_tree,
    root_attribute,
    tree_depth,
    tree_to_text,
)
from problem import (
    DEFAULT_DATA_PATH,
    TARGET_COLUMN,
    VARIANTS,
    load_data,
    prepare_variant,
    split_train_test,
)


def save_log(text, path="monitor.log"):
    # Zapis pojedynczej informacji do pliku monitorujacego.
    with open(path, "a", encoding="utf-8") as file:
        file.write(text + "\n")


def save_tree_to_file(tree_text, criterion, variant_name):
    # Zapis zbudowanego drzewa do osobnego pliku tekstowego.
    folder = Path("wyniki_drzew")
    folder.mkdir(exist_ok=True)
    file_path = folder / f"drzewo_{criterion}_{variant_name}.txt"
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(tree_text + "\n")
    return str(file_path)


def accuracy(tree, test_data, target):
    # Prosty pomiar skutecznosci klasyfikacji.
    correct = 0
    for _, row in test_data.iterrows():
        sample = row.to_dict()
        true_class = sample.pop(target)
        predicted = classify(tree, sample)
        if predicted == true_class:
            correct += 1

    if len(test_data) == 0:
        return 0

    return correct / len(test_data)


def run_single(path, criterion, variant_name):
    # Jedno uruchomienie dla wybranego kryterium i wariantu danych.
    data = load_data(path)
    data, attributes, removed = prepare_variant(data, TARGET_COLUMN, variant_name)
    train_data, test_data = split_train_test(data, TARGET_COLUMN)
    tree = build_tree(train_data, attributes, TARGET_COLUMN, criterion)
    result_accuracy = accuracy(tree, test_data, TARGET_COLUMN)
    tree_text = tree_to_text(tree)
    tree_file_path = save_tree_to_file(tree_text, criterion, variant_name)
    nodes = count_nodes(tree)
    depth = tree_depth(tree)
    root = root_attribute(tree)

    print()
    print("Kryterium:", criterion)
    print("Wariant:", variant_name)
    print("Usuniete atrybuty:", removed if removed else "brak")
    print("Uzyte atrybuty:", attributes)
    print("Liczba probek treningowych:", len(train_data))
    print("Liczba probek testowych:", len(test_data))
    print("Korzen drzewa:", root)
    print("Liczba wezlow:", nodes)
    print("Glebokosc:", depth)
    print("Accuracy:", round(result_accuracy * 100, 2), "%")
    print("Plik drzewa:", tree_file_path)
    print("Drzewo:")
    print(tree_text)
    print()

    log_text = (
        f"{datetime.now()} | kryterium={criterion} | wariant={variant_name} | "
        f"usuniete={removed if removed else 'brak'} | "
        f"atrybuty={attributes} | train={len(train_data)} | test={len(test_data)} | "
        f"korzen={root} | wezly={nodes} | glebokosc={depth} | "
        f"accuracy={round(result_accuracy * 100, 2)}% | plik_drzewa={tree_file_path}"
    )
    save_log(log_text)
    return {
        "criterion": criterion,
        "variant": variant_name,
        "removed": removed if removed else ["brak"],
        "attributes": attributes,
        "train_count": len(train_data),
        "test_count": len(test_data),
        "root": root,
        "nodes": nodes,
        "depth": depth,
        "accuracy": round(result_accuracy * 100, 2),
    }


def compare_all(path):
    # Porownanie obu kryteriow dla wszystkich przygotowanych wariantow.
    criteria = ["info_gain", "gain_ratio"]
    results = []

    for variant in VARIANTS:
        for criterion in criteria:
            results.append(run_single(path, criterion, variant))

    print_summary(results)


def print_summary(results):
    # Krotkie tabelaryczne podsumowanie do terminala i logu.
    print("=== Podsumowanie porownania ===")
    print("wariant | kryterium | accuracy | korzen | wezly | glebokosc")
    for result in results:
        print(
            f"{result['variant']} | {result['criterion']} | {result['accuracy']}% | "
            f"{result['root']} | {result['nodes']} | {result['depth']}"
        )
    print()

    save_log("=== Podsumowanie porownania ===")
    for result in results:
        save_log(
            f"wariant={result['variant']} | kryterium={result['criterion']} | "
            f"accuracy={result['accuracy']}% | korzen={result['root']} | "
            f"wezly={result['nodes']} | glebokosc={result['depth']}"
        )


def menu():
    data_path = DEFAULT_DATA_PATH

    while True:
        print("=== Projekt U3 - Sterowanie ===")
        print("1. Ustaw sciezke do danych")
        print("2. Zbuduj drzewo dla jednego kryterium")
        print("3. Porownaj oba kryteria i wszystkie warianty")
        print("4. Zakoncz")
        choice = input("Wybor: ")

        if choice == "1":
            new_path = input("Podaj sciezke do pliku CSV: ").strip()
            if new_path:
                if Path(new_path).exists():
                    data_path = new_path
                else:
                    print("Podana sciezka nie istnieje.")
            print("Aktualna sciezka:", data_path)
            print()

        elif choice == "2":
            criterion = input("Podaj kryterium (info_gain/gain_ratio): ").strip()
            variant = input(
                "Podaj wariant (full / without_crossed_out / without_element_count / without_shape_and_border): "
            ).strip()

            if criterion == "":
                criterion = "info_gain"
            if variant == "":
                variant = "full"

            if criterion not in ["info_gain", "gain_ratio"]:
                print("Bledne kryterium. Wpisz info_gain albo gain_ratio.")
                print()
                continue

            if variant not in VARIANTS:
                print("Bledny wariant. Dostepne warianty:", ", ".join(VARIANTS))
                print()
                continue

            if not Path(data_path).exists():
                print("Plik z danymi nie istnieje.")
                print()
                continue

            run_single(data_path, criterion, variant)

        elif choice == "3":
            if not Path(data_path).exists():
                print("Plik z danymi nie istnieje.")
                print()
                continue
            compare_all(data_path)

        elif choice == "4":
            print("Koniec programu.")
            break

        else:
            print("Niepoprawny wybor.")
            print()
