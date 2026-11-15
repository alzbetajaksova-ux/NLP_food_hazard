"""Spoločné funkcie pre 2. etapu: načítanie dát, tokenizácia, oficiálne skóre
a formáty predikcií a kandidátov. Všetky skripty a notebooky 2. etapy ich importujú,
aby každý počítal a ukladal výsledky rovnako."""
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SEED = 42
LABELY = ["hazard-category", "product-category", "hazard", "product"]
TOP_K = 10

KOREN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(KOREN, "data")
VYSLEDKY = os.path.join(KOREN, "results", "etapa2")
PREDIKCIE = os.path.join(VYSLEDKY, "predikcie")
KANDIDATI = os.path.join(VYSLEDKY, "kandidati")

URL = ("https://raw.githubusercontent.com/food-hazard-detection-semeval-2025/"
       "food-hazard-detection-semeval-2025.github.io/main/data/")
SPLITY = {"train": "incidents_train", "valid": "incidents_valid", "test": "incidents_test"}


def nacitaj(split):
    """Vráti DataFrame pre 'train', 'valid' alebo 'test' (z data/, inak z repozitára úlohy)."""
    subor = os.path.join(DATA, SPLITY[split] + ".csv")
    if os.path.exists(subor):
        return pd.read_csv(subor)
    return pd.read_csv(URL + SPLITY[split] + ".csv")


TOKEN_REGEX = re.compile(r"\w+|[^\w\s]")


def tokenizuj(text):
    """Rovnaký tokenizér ako baseline v 1. etape."""
    return TOKEN_REGEX.findall(text.lower())


def vstup(data):
    """Vstup baseline: titulok + text."""
    return data["title"] + " " + data["text"]


def oficialne_skore(hazard_true, product_true, hazard_pred, product_pred):
    """Oficiálne skóre SemEval-2025 Task 9 (logika compute_score organizátorov):
    (macro-F1 hazard + macro-F1 produkt na správach so správnym hazardom) / 2."""
    hazard_true, product_true = np.asarray(hazard_true), np.asarray(product_true)
    hazard_pred, product_pred = np.asarray(hazard_pred), np.asarray(product_pred)
    f1_hazard = f1_score(hazard_true, hazard_pred, average="macro", zero_division=0)
    ok = hazard_pred == hazard_true
    f1_product = f1_score(product_true[ok], product_pred[ok], average="macro", zero_division=0)
    return (f1_hazard + f1_product) / 2


def skore(data, pred):
    """ST1, ST2 a macro-F1 pre každý štítok. pred = DataFrame alebo dict so stĺpcami LABELY."""
    vysl = {
        "ST1": oficialne_skore(data["hazard-category"], data["product-category"],
                               pred["hazard-category"], pred["product-category"]),
        "ST2": oficialne_skore(data["hazard"], data["product"], pred["hazard"], pred["product"]),
    }
    for lab in LABELY:
        vysl[lab] = f1_score(data[lab], pred[lab], average="macro", zero_division=0)
    return vysl


# ---------- formáty ----------

def cesta_predikcii(system, split):
    return os.path.join(PREDIKCIE, f"{system}_{split}.csv")


def cesta_kandidatov(system, split):
    return os.path.join(KANDIDATI, f"{system}_{split}.csv")


def uloz_predikcie(pred, system, split):
    """pred: DataFrame so stĺpcami id + LABELY, jeden riadok na správu, v poradí splitu."""
    data = nacitaj(split)
    pred = pred[["id"] + LABELY].copy()
    assert len(pred) == len(data), "počet riadkov nesedí so splitom"
    assert (pred["id"].values == np.arange(len(data))).all(), "id musí byť 0..n-1 v poradí splitu"
    assert not pred[LABELY].isna().any().any(), "predikcie nesmú byť prázdne"
    os.makedirs(PREDIKCIE, exist_ok=True)
    pred.to_csv(cesta_predikcii(system, split), index=False)
    return cesta_predikcii(system, split)


def nacitaj_predikcie(system_alebo_cesta, split=None):
    cesta = system_alebo_cesta if split is None else cesta_predikcii(system_alebo_cesta, split)
    return pd.read_csv(cesta, dtype=str, keep_default_na=False).assign(id=lambda d: d["id"].astype(int))


def uloz_kandidatov(kand, system, split):
    """kand: DataFrame so stĺpcami id, stitok, poradie (1 = najlepší), trieda, skore.
    Pre každú správu a každý štítok TOP_K riadkov (ak model nevie skóre, skore = NaN)."""
    kand = kand[["id", "stitok", "poradie", "trieda", "skore"]].copy()
    assert set(kand["stitok"]) <= set(LABELY)
    os.makedirs(KANDIDATI, exist_ok=True)
    kand.to_csv(cesta_kandidatov(system, split), index=False)
    return cesta_kandidatov(system, split)


def nacitaj_kandidatov(system_alebo_cesta, split=None):
    cesta = system_alebo_cesta if split is None else cesta_kandidatov(system_alebo_cesta, split)
    kand = pd.read_csv(cesta, dtype=str, keep_default_na=False)
    kand["id"] = kand["id"].astype(int)
    kand["poradie"] = kand["poradie"].astype(int)
    kand["skore"] = pd.to_numeric(kand["skore"], errors="coerce")
    return kand
