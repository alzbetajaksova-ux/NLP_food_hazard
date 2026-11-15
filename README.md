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

## 2. etapa: spoločné formáty a skripty (rozpracované)

Všetky systémy 2. etapy ukladajú výstupy v rovnakom formáte a vyhodnocujú sa jedným skriptom, aby sa dali férovo porovnať s baseline (E0).

```
src/
├── spolocne.py        načítanie dát, tokenizér, oficiálne skóre, ukladanie a načítanie formátov
├── vyhodnot.py        skóre, bootstrap intervaly, párové porovnanie, recall@k kandidátov
├── e0_svm_export.py   E0: predikcie a top-10 kandidátov baseline SVM (valid aj test)
└── e3_hierarchia.py   E3: hierarchia hazard → kategória a produkt z titulku nad ľubovoľným systémom
results/etapa2/
├── predikcie/<systém>_<split>.csv
└── kandidati/<systém>_<split>.csv
```

**Predikcie** (`results/etapa2/predikcie/<systém>_<split>.csv`): jeden riadok na správu, stĺpce `id, hazard-category, product-category, hazard, product`. `id` je poradie správy v splite (0 až n − 1).

**Kandidáti** (`results/etapa2/kandidati/<systém>_<split>.csv`): pre každú správu a každý štítok 10 riadkov, stĺpce `id, stitok, poradie, trieda, skore`. `poradie` 1 je najpravdepodobnejšia trieda; `skore` je skóre modelu (pri SVM `decision_function`, pri transformeri pravdepodobnosť), väčšie = lepšie.

Názov systému = číslo experimentu a model, napríklad `E0_svm`, `E2_roberta_s42`, `E3_svm`, `E4_llm_svm`. Pri ukladaní z Pythonu použite `uloz_predikcie` a `uloz_kandidatov` zo `src/spolocne.py`, ktoré formát skontrolujú.

```bash
python src/e0_svm_export.py                                   # E0, asi pol minúty
python src/vyhodnot.py --split valid E0_svm --kandidati E0_svm  # skóre a recall@k na valid
python src/e3_hierarchia.py --zaklad E0_svm                   # E3 nad SVM, vypíše abláciu na valid
python src/vyhodnot.py --split valid E3_svm --porovnaj E0_svm --bootstrap 1000
```

Pravidlá: nastavenia sa vyberajú len na `valid`. Test sa vyhodnotí jediný raz na konci pre všetky zmrazené systémy (`--split test --povolit-test`); bez prepínača skript test odmietne.

Baseline E0 na valid (kontrola): ST1 0,736, ST2 0,416. Správna trieda medzi top-10 kandidátmi: hazard 96,6 %, produkt 67,6 % (na 1. mieste 82,7 % a 40,9 %).

**E3 nad SVM (valid, k = 5)**: hierarchia zvýši ST1 z 0,736 na 0,747, produkt z titulku zvýši ST2 z 0,416 na 0,426 (párový bootstrap: +0,011 [+0,000; +0,020], lepší v 98 % výberov). Kandidáti E3 majú správny produkt v top-10 v 71,0 % správ (E0: 67,6 %). Hodnota k bola vybraná na valid z možností 3, 5 a 10, preto je výsledok na valid mierne optimistický; potvrdí ho až test na konci. Pri kandidátoch E3 je na 1. mieste produkt z titulku, preto sa top-1 kandidát nemusí zhodovať s predikciou.
