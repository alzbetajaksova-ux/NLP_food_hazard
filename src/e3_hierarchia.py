"""E3: druhý krok bez LLM nad ľubovoľným systémom (predvolene E0_svm).

1. Hierarchia: hazard-category sa odvodí z predikovaného hazardu (v train patrí každý hazard,
   okrem jedného, do jedinej kategórie), takže si hazard a jeho kategória nikdy neodporujú.
2. Produkt z titulku: ak titulok obsahuje názov produktu známeho z train (najdlhšia zhoda,
   celé slová) a model ho má medzi top-K kandidátmi, predikciou sa stane tento produkt.
3. Kandidáti pre LLM (E4): produkt z titulku sa dá na 1. miesto, potom kandidáti modelu (spolu 10).

Výstup: results/etapa2/predikcie/E3_<zaklad>_<split>.csv a results/etapa2/kandidati/E3_<zaklad>_<split>.csv
K sa vyberá na valid (predvolene 5). Na valid vypíše abláciu: základ, +hierarchia, +titulok, oboje.

Beh: python src/e3_hierarchia.py --zaklad E0_svm
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spolocne import (KOREN, LABELY, TOP_K, nacitaj, nacitaj_kandidatov,  # noqa: E402
                      nacitaj_predikcie, skore, uloz_kandidatov, uloz_predikcie)


def mapa_hazard_kategoria(train):
    """hazard -> najčastejšia hazard-category v train."""
    return train.groupby("hazard")["hazard-category"].agg(lambda s: s.value_counts().index[0])


class ProduktZTitulku:
    """Hľadá v titulku názvy produktov z train (bez ohľadu na veľkosť písmen, celé slová)."""

    def __init__(self, train):
        produkty = train["product"]
        self.povodny = produkty.groupby(produkty.str.lower()).agg(lambda s: s.value_counts().index[0])
        nazvy = sorted(self.povodny.index, key=len, reverse=True)
        self.regex = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in nazvy) + r")\b")

    def najdi(self, titulok):
        zhody = self.regex.findall(titulok.lower())
        return self.povodny[max(zhody, key=len)] if zhody else ""


def uprav(pred, kand, data, mapa, titulky, k, hierarchia=True, titulok=True):
    """Vráti nové predikcie a kandidátov podľa zapnutých krokov."""
    pred = pred.copy()
    if hierarchia:
        odvodene = pred["hazard"].map(mapa)
        pred["hazard-category"] = odvodene.fillna(pred["hazard-category"])

    kp = kand[kand["stitok"] == "product"].sort_values(["id", "poradie"])
    top = {i: g["trieda"].tolist() for i, g in kp.groupby("id")}
    zhody = [titulky.najdi(t) for t in data["title"]]
    if titulok:
        pred["product"] = [z if z and z in top[i][:k] else p
                           for i, (z, p) in enumerate(zip(zhody, pred["product"]))]

    # kandidáti produktu: zhoda z titulku na 1. mieste, potom model, spolu TOP_K
    skore_modelu = {(r.id, r.trieda): r.skore for r in kp.itertuples()}
    riadky = []
    for i in range(len(data)):
        zoznam = ([zhody[i]] if zhody[i] else []) + [c for c in top[i] if c != zhody[i]]
        for poradie, trieda in enumerate(zoznam[:TOP_K], start=1):
            riadky.append((i, "product", poradie, trieda, skore_modelu.get((i, trieda), np.nan)))
    nove = pd.concat([kand[kand["stitok"] != "product"],
                      pd.DataFrame(riadky, columns=["id", "stitok", "poradie", "trieda", "skore"])])
    return pred, nove.sort_values(["stitok", "id", "poradie"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zaklad", default="E0_svm", help="systém, ktorého predikcie a kandidátov upravíme")
    ap.add_argument("--k", type=int, default=5, help="produkt z titulku sa použije, ak je v top-k modelu")
    a = ap.parse_args()

    train = nacitaj("train")
    mapa, titulky = mapa_hazard_kategoria(train), ProduktZTitulku(train)
    system = "E3_" + (a.zaklad.split("_", 1)[1] if "_" in a.zaklad else a.zaklad)

    # ablácia na valid
    valid = nacitaj("valid")
    pred, kand = nacitaj_predikcie(a.zaklad, "valid"), nacitaj_kandidatov(a.zaklad, "valid")
    riadky = []
    for meno, h, t in [("základ", False, False), ("+ hierarchia", True, False),
                       ("+ produkt z titulku", False, True), ("oboje (E3)", True, True)]:
        p, _ = uprav(pred, kand, valid, mapa, titulky, a.k, h, t)
        s = skore(valid, p)
        riadky.append({"variant": meno, **{x: round(s[x], 4) for x in ["ST1", "ST2"] + LABELY}})
    print(f"Ablácia na valid ({a.zaklad}, k = {a.k}):")
    print(pd.DataFrame(riadky).to_string(index=False))

    for split in ["valid", "test"]:
        data = nacitaj(split)
        p, k = uprav(nacitaj_predikcie(a.zaklad, split), nacitaj_kandidatov(a.zaklad, split),
                     data, mapa, titulky, a.k)
        c1, c2 = uloz_predikcie(p, system, split), uloz_kandidatov(k, system, split)
        print("uložené:", os.path.relpath(c1, KOREN), "|", os.path.relpath(c2, KOREN))


if __name__ == "__main__":
    main()
