Projekt U3

Budowa drzewa decyzyjnego dla identyfikacji znaków drogowych

Projekt został wykonany w ramach przedmiotu Metody Sztucznej Inteligencji.
Celem programu jest budowa drzewa decyzyjnego służącego do klasyfikacji znaków drogowych na podstawie ich cech, takich jak kształt, kolor tła, kolor obramowania, kolor symbolu, liczba elementów oraz informacja o przekreśleniu.

W projekcie zaimplementowano dwa kryteria wyboru najlepszego atrybutu:
- InfoGain – zysk informacji,
- GainRatio – iloraz zysku informacji.

Pliki

- metoda.py – implementacja algorytmu drzewa decyzyjnego,
- problem.py – wczytywanie danych i przygotowanie wariantów eksperymentu,
- sterowanie.py – menu tekstowe, uruchamianie eksperymentów i zapis wyników,
- main.py – punkt wejścia programu,
- dane/road_signs.csv – przykładowy zbiór danych.

Dane wejściowe

Każdy rekord w pliku road_signs.csv opisuje jeden znak drogowy za pomocą cech:
- shape,
- background_color,
- border_color,
- symbol_color,
- element_count,
- crossed_out.

Zmienna decyzyjna:
- sign_class.

Dostępne warianty
Program obsługuje następujące warianty danych:
- full,
- without_crossed_out,
- without_element_count,
- without_shape_and_border.

Pozwala to sprawdzić, jak usunięcie wybranych cech wpływa na strukturę drzewa i skuteczność klasyfikacji.

Uruchomienie

python main.py

Po uruchomieniu program wyświetla menu tekstowe i pozwala:
- wybrać plik z danymi,
- uruchomić pojedynczy eksperyment,
- wykonać pełne porównanie obu kryteriów.

Wyniki

Program zapisuje:
- przebieg działania do pliku monitor.log,
- wygenerowane drzewa do folderu wyniki_drzew.

W aktualnej wersji użyto seed = 42, aby wyniki były powtarzalne.

Wymagania:
- Python 3
- pandas
- numpy

Instalacja bibliotek:
'pip install pandas numpy'

Podsumowanie:
Program realizuje budowę drzewa decyzyjnego dla klasyfikacji znaków drogowych i umożliwia porównanie działania metod InfoGain oraz GainRatio dla różnych wariantów danych.
