# NLP projekt – Food Hazard Detection (SemEval 2025 Task 9)

Tím: Alexandra Domonkošová, Kristína Gvozdiaková, Alžbeta Jakšová

Baseline TF-IDF + logistická regresia a lineárny SVM pre obe podúlohy:
- **ST1:** `hazard-category` (10 tried) a `product-category` (22 tried)
- **ST2:** `hazard` (128 tried) a `product` (1 022 tried v train)

Výsledky na testovacej množine (vstup titulok + text, váhy tried `balanced`):

| Model | ST1 | ST2 |
| --- | --- | --- |
| TF-IDF + SVM | 0,723 | 0,410 |
| TF-IDF + LR | 0,717 | 0,406 |
| Majoritná trieda | 0,035 | 0,001 |

Podrobná tabuľka vrátane modelov len na titulku je v `results/summary.csv`.

## Štruktúra

```
NLP_food_hazard/
├── README.md
├── requirements.txt
├── dokumentacia_1etapa.docx          správa k 1. etape
├── priloha_pouzitie_LLM.docx         príloha o použití LLM a AI agentov
├── data/
│   ├── incidents_train.csv, incidents_valid.csv, incidents_test.csv
│   └── statistics_train.ipynb        starší pomocný prehľad trénovacích dát (spúšťa sa z data/)
├── notebooks/
│   ├── baseline_food_hazard.ipynb    trénovanie, výber nastavení na validácii, test, uloženie výsledkov a predikcií
│   └── eda_chybova_analyza.ipynb     analýza dát, chybová analýza, bootstrap intervaly spoľahlivosti
└── results/                          všetko vytvoria notebooky
    ├── summary.csv                   skóre na teste (baseline)
    ├── predikcie_test.csv            predikcie SVM a LR na teste (baseline, číta ich EDA notebook)
    ├── bootstrap.csv                 95 % intervaly a rozdiel SVM − LR (EDA)
    ├── chyby_na_komentar.csv         20 chýb s ručným komentárom v stĺpci „dôvod“ (EDA)
    └── figures/                      6 obrázkov do správy (EDA)

```

## Spustenie

Treba **Python 3.11 alebo novší** (overené na 3.13); verzie v `requirements.txt` sa na Python 3.10 nenainštalujú. GPU netreba.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace baseline_food_hazard.ipynb
jupyter nbconvert --to notebook --execute --inplace eda_chybova_analyza.ipynb
```

**Poradie je dôležité:** EDA notebook netrénuje modely, načíta predikcie z `results/predikcie_test.csv`, ktoré uloží baseline notebook.

V Jupyteri / datalabe: otvoriť notebook v priečinku `notebooks/` a spustiť **Run → Run All Cells**, najprv baseline, potom EDA. Dáta sa načítajú z `data/`; ak tam nie sú, notebooky si ich stiahnu z repozitára úlohy.

Čas behu závisí od procesora: baseline približne 5–10 minút (najdlhšie trvá logistická regresia pre 1 022 produktov), EDA približne 1 minútu. Stĺpec `čas [s]` v `summary.csv` je meranie z posledného behu a na inom stroji bude iný; skóre sa nemenia.

## Reprodukovateľnosť

- Rozdelenie train / valid / test je oficiálne od organizátorov, nič sa nedelí náhodne.
- Všetky náhodné kroky majú `SEED = 42` (vrátane `random_state` pri SVM aj LR a výberu chýb a bootstrapu).
- Slovník a IDF sa počítajú iba z trénovacích dát; finálne modely sa trénujú iba na train (valid sa nepridáva).
- Nastavenia (vstup, váženie tried) sa vyberajú iba na validačnej množine podľa ST1, test sa vyhodnotí raz.
- Oficiálne skóre = (macro-F1 hazard + macro-F1 produkt na hláseniach so správnym hazardom) / 2. Funkcia `oficialne_skore` zodpovedá funkcii `compute_score` z oficiálneho kódu organizátorov (`code/` v [repozitári úlohy](https://github.com/food-hazard-detection-semeval-2025/food-hazard-detection-semeval-2025.github.io/tree/main/code)).

## Dáta a licencia

Dáta sú z [repozitára úlohy](https://github.com/food-hazard-detection-semeval-2025/food-hazard-detection-semeval-2025.github.io/tree/main/data). Podľa stránky úlohy je úplný dataset zverejnený pod licenciou **Creative Commons BY-NC-SA 4.0** na [Zenode](https://zenodo.org/doi/10.5281/zenodo.10820657): pri použití treba uviesť autorov, nesmie sa použiť komerčne a odvodené diela sa musia šíriť pod rovnakou licenciou. Kód organizátorov v ich repozitári má licenciu GPL-3.0.

Citácia úlohy a datasetu: Randl, K. et al. (2025). *SemEval-2025 Task 9: The Food Hazard Detection Challenge.* arXiv:2503.19800.
