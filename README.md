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
├── requirements.txt                  závislosti 1. etapy (a CPU časti 2. etapy)
├── requirements_etapa2.txt           navyše pre transformer (torch, transformers)
├── dokumentacia_1etapa.docx          správa k 1. etape
├── dokumentacia_2etapa_kostra.docx   rozpracovaná správa k 2. etape (kto píše ktorú sekciu)
├── priloha_pouzitie_LLM.docx         príloha o použití LLM a AI agentov (1. etapa)
├── data/
│   ├── incidents_train.csv, incidents_valid.csv, incidents_test.csv
│   └── statistics_train.ipynb        starší pomocný prehľad trénovacích dát (spúšťa sa z data/)
├── notebooks/
│   ├── baseline_food_hazard.ipynb    trénovanie, výber nastavení na validácii, test, uloženie výsledkov a predikcií
│   ├── eda_chybova_analyza.ipynb     analýza dát, chybová analýza, bootstrap intervaly spoľahlivosti
│   └── etapa2_transformer_colab.ipynb  2. etapa: doladenie transformera na Google Colab s GPU
├── src/                              2. etapa: skripty (pozri nižšie)
├── docs/etapa2/                      2. etapa: zápisník experimentov, návody, taxonómia, pravidlá, šablóny
└── results/                          1. etapa: všetko vytvoria notebooky; 2. etapa: results/etapa2/
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

## 2. etapa (rozpracované)

Vlastné riešenie: **pre každý štítok použiť model, ktorý je preň najlepší (doladený transformer alebo lineárny model s dôrazom na titulok), behy rovnakého modelu spojiť do ensemble a pre jemné štítky nechať LLM vybrať odpoveď z 10 kandidátov.** Všetky experimenty, výsledky na valid a rozhodnutia sú v `docs/etapa2/zapisnik_experimentov.md`.

### Výsledky na valid (17. 11. 2026)

| Systém | Čo to je | ST1 | ST2 |
| --- | --- | --- | --- |
| `E0_svm` | baseline SVM z 1. etapy | 0,736 | 0,416 |
| `E0_svmvyber` | SVM so znakovými n-gramami titulku, výber po štítkoch | 0,777 | 0,447 |
| `E2kens` | + hazard-category z RoBERTa (ensemble 3 semienok) | 0,802 | 0,447 |
| `E12kens` | + hazard a product z RoBERTa s dlhším tréningom (ensemble 3 semienok) | **0,802** | **0,495** |

`E12kens` je oproti `E0_svm` lepší o +0,066 v ST1 ([+0,034; +0,104]) a +0,079 v ST2 ([+0,047; +0,112]). Pravidlá kombinácie boli zvolené na valid, výsledok preto potvrdí až jediný beh na teste. Výber pomocou LLM (E4) a test zatiaľ chýbajú.

### Štruktúra

```
src/
├── spolocne.py           načítanie dát, tokenizér, oficiálne skóre, ukladanie a kontrola formátov
├── vyhodnot.py           skóre, bootstrap, párové porovnanie, recall@k kandidátov; test len s --povolit-test
├── e0_svm_export.py      E0_svm (baseline) a s --titulok E0_svmtit (znakové n-gramy titulku)
├── vyber_po_stitkoch.py  pre každý štítok lepší systém podľa valid (E0_svmvyber)
├── e12_transformer.py    doladenie RoBERTa pre zvolené štítky (E1, E2, E1b; GPU)
├── ensemble.py           ensemble behov s rôznymi semienkami (súčet pravdepodobností)
├── kombinacia.py         pevná kombinácia dvoch systémov po štítkoch (E2kens, E12kens)
├── zostav_systemy.py     zostaví všetky CPU systémy z uložených výstupov v správnom poradí
├── e3_hierarchia.py      ablácia: hierarchia hazard → kategória a produkt z titulku
├── e4_llm.py             E4: LLM vyberie hazard a produkt z top-10 kandidátov (osoba 2)
├── chyby_vzorka.py       vzorka chýb na valid pre dve anotátorky
└── zhoda.py              Cohenova kappa, matica zhody a zoznam nezhôd
docs/etapa2/
├── zapisnik_experimentov.md          experimenty, výsledky na valid a rozhodnutia
├── navod_gpu.md                      Google Colab a Kaggle pre transformer
├── taxonomia_chyb.md                 kategórie chýb a postup anotácie (zapísané pred anotáciou)
├── anotacne_pravidla_novych_dat.md   pravidlá pre vlastný dataset novších správ (osoba 3)
├── datasheet_sablona.md              šablóna datasheetu pre vlastný dataset
└── protokol_reprodukovatelnosti.md   šablóna protokolu nezávislej reprodukovateľnosti
results/etapa2/
├── predikcie/<systém>_<split>.csv    predikcie (valid aj test)
├── kandidati/<systém>_<split>.csv    top-10 tried so skóre
├── logy/<systém>.json                nastavenia a časy behov transformera, voľba pri výbere po štítkoch
└── chyby/                            vzorka 200 chýb E12kens na anotáciu (anotacia_A.csv, anotacia_B.csv, kluc.csv)
```

V gite sú výstupy transformera (behy `E1_`, `E2_`, `E1b_robertabase_s42/43/44`), ktoré sa nedajú rýchlo zopakovať, a výsledné systémy `E0_svm`, `E0_svmvyber`, `E2kens` a `E12kens`. Medzikroky (`E0_svmtit`, ensemble, ablácie E3) vytvorí `src/zostav_systemy.py` a sú v `.gitignore`.

### Formáty

- **Predikcie:** jeden riadok na správu, stĺpce `id, hazard-category, product-category, hazard, product`; `id` je poradie správy v splite (0 až n − 1).
- **Kandidáti:** pre každú správu a štítok 10 riadkov, stĺpce `id, stitok, poradie, trieda, skore`; `poradie` 1 je najlepšia trieda, `skore` je pri SVM `decision_function`, pri transformeri pravdepodobnosť (väčšie = lepšie).
- Na ukladanie slúžia `uloz_predikcie` a `uloz_kandidatov` zo `src/spolocne.py`, ktoré formát skontrolujú.

### Spustenie

```bash
pip install -r requirements.txt
python src/zostav_systemy.py                    # všetky CPU systémy, asi 1 minúta
python src/vyhodnot.py --split valid E0_svm E0_svmvyber E2kens E12kens --porovnaj E0_svm --bootstrap 1000
```

Transformer (GPU): `notebooks/etapa2_transformer_colab.ipynb` na Google Colab, návod v `docs/etapa2/navod_gpu.md`. Lokálne `pip install -r requirements_etapa2.txt` a `python src/e12_transformer.py --help`.

Výber pomocou LLM (osoba 2): `python src/e4_llm.py --backend skuska` overí celú cestu bez LLM; so skutočným modelom nastav `LLM_BASE_URL` a `LLM_API_KEY` (API kompatibilné s OpenAI, aj lokálna Ollama) a spusti najprv s `--limit 50`. Odpovede sa ukladajú do `results/etapa2/llm_cache/`, takže opakovaný beh dá rovnaký výsledok bez volaní.

### Pravidlá

- Nastavenia, varianty a pravidlá kombinácie sa vyberajú len na `valid`.
- Test sa vyhodnotí jediný raz na konci pre všetky zmrazené systémy (`--split test --povolit-test`); bez prepínača ho skript odmietne.
- Transformer: semienka 42, 43, 44; vo výsledných systémoch ensemble semienok.
