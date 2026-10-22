# NLP projekt – Food Hazard Detection (SemEval 2025 Task 9)

Tím: Alexandra Domonkošová, Kristína Gvozdiaková, Alžbeta Jakšová

Baseline TF-IDF + logistická regresia a lineárny SVM pre obe podúlohy:
- **ST1:** `hazard-category` (10 tried) a `product-category` (22 tried)
- **ST2:** `hazard` (128 tried) a `product` (1 022 tried v train)

## Štruktúra

| Súbor | Obsah |
| --- | --- |
| `notebooks/baseline_food_hazard.ipynb` | celý postup: dáta, metrika, TF-IDF, výber nastavení na validácii, test, podklady pre analýzu chýb |
| `notebooks/results/summary.csv` | výsledky na teste (vytvorí notebook) |
| `notebooks/results/chyby_hazard_category.csv` | 20 chýb na okomentovanie (vytvorí notebook) |
| `data/` | `incidents_train.csv`, `incidents_valid.csv`, `incidents_test.csv` a `statistics_train.ipynb` (štatistiky trénovacích dát) |
| `requirements.txt` | verzie knižníc, s ktorými boli výsledky vypočítané |

Dáta sú z [repozitára úlohy](https://github.com/food-hazard-detection-semeval-2025/food-hazard-detection-semeval-2025.github.io/tree/main/data) (licencia CC BY-NC-SA 4.0). Ak v `data/` chýbajú, notebook si ich stiahne.

## Spustenie

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace baseline_food_hazard.ipynb
```

Alebo v datalabe / Jupyteri: otvoriť `notebooks/baseline_food_hazard.ipynb` a spustiť **Run → Run All Cells**.

Treba **Python 3.11 alebo novší** (overené na 3.13); verzie v `requirements.txt` sa na Python 3.10 nenainštalujú.

Beh trvá na CPU približne 5–7 minút v čistom prostredí podľa `requirements.txt`. V iných prostrediach (napr. Anaconda so staršími knižnicami) môže trvať aj vyše 10 minút, výsledky sú rovnaké. GPU netreba.

## Reprodukovateľnosť

- Rozdelenie train / valid / test je oficiálne od organizátorov, nič sa nedelí náhodne.
- Všetky náhodné kroky majú `SEED = 42`.
- Slovník a IDF sa počítajú iba z trénovacích dát.
- Nastavenia (vstup, váženie tried) sa vyberajú iba na validačnej množine, test sa vyhodnotí raz.
- Oficiálne skóre = (macro-F1 hazard + macro-F1 produkt na hláseniach so správnym hazardom) / 2.
