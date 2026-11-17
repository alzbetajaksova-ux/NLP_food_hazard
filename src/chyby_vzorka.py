"""Vzorka chýb na ručnú anotáciu podľa docs/etapa2/taxonomia_chyb.md.

Vyberie náhodne (SEED) chybné predikcie štítkov hazard a product na valid a vytvorí:
  results/etapa2/chyby/anotacia_A.csv   pre osobu 2
  results/etapa2/chyby/anotacia_B.csv   pre osobu 3 (tie isté položky v inom poradí)
  results/etapa2/chyby/kluc.csv         id položiek, systém a pomocné stĺpce

Beh: python src/chyby_vzorka.py --system E12kens --pocet 200
Existujúce anotačné súbory sa neprepíšu (ochrana proti strate ručnej práce), iba s --prepis.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spolocne import KOREN, SEED, VYSLEDKY, nacitaj, nacitaj_predikcie  # noqa: E402

PRIECINOK = os.path.join(VYSLEDKY, "chyby")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True)
    ap.add_argument("--pocet", type=int, default=200)
    ap.add_argument("--min-hazard", type=int, default=60)
    ap.add_argument("--prepis", action="store_true")
    a = ap.parse_args()

    os.makedirs(PRIECINOK, exist_ok=True)
    ciel = [os.path.join(PRIECINOK, f) for f in ["anotacia_A.csv", "anotacia_B.csv", "kluc.csv"]]
    if any(os.path.exists(c) for c in ciel) and not a.prepis:
        sys.exit("Anotačné súbory už existujú. Ak ich naozaj chceš vytvoriť znova, pridaj --prepis.")

    valid, train = nacitaj("valid"), nacitaj("train")
    pred = nacitaj_predikcie(a.system, "valid").sort_values("id").reset_index(drop=True)
    rng = np.random.default_rng(SEED)

    chyby = {lab: np.where(pred[lab].values != valid[lab].values)[0] for lab in ["hazard", "product"]}
    spolu = len(chyby["hazard"]) + len(chyby["product"])
    n_h = min(len(chyby["hazard"]), max(a.min_hazard, round(a.pocet * len(chyby["hazard"]) / spolu)))
    n_p = min(len(chyby["product"]), a.pocet - n_h)
    vyber = [("hazard", i) for i in rng.choice(chyby["hazard"], n_h, replace=False)] + \
            [("product", i) for i in rng.choice(chyby["product"], n_p, replace=False)]

    pocty = {lab: train[lab].value_counts() for lab in ["hazard", "product"]}
    riadky = []
    for poradie, (lab, i) in enumerate(vyber):
        pravda, predikcia = valid[lab][i], pred[lab][i]
        riadky.append({
            "polozka": poradie, "id_spravy": int(i), "stitok": lab,
            "titulok": valid["title"][i],
            "text_zaciatok": re.sub(r"\s+", " ", valid["text"][i])[:600],
            "pravda": pravda, "predikcia": predikcia,
            "pocet_pravdy_v_train": int(pocty[lab].get(pravda, 0)),
            "podretazec": int(pravda.lower() in predikcia.lower() or predikcia.lower() in pravda.lower()),
            "kategoria": "", "poznamka": "",
        })
    tab = pd.DataFrame(riadky)
    stlpce = ["polozka", "id_spravy", "stitok", "titulok", "text_zaciatok", "pravda", "predikcia",
              "pocet_pravdy_v_train", "podretazec", "kategoria", "poznamka"]
    tab[stlpce].sample(frac=1, random_state=SEED).to_csv(ciel[0], index=False)
    tab[stlpce].sample(frac=1, random_state=SEED + 1).to_csv(ciel[1], index=False)
    tab[["polozka", "id_spravy", "stitok", "pravda", "predikcia", "pocet_pravdy_v_train", "podretazec"]] \
        .assign(system=a.system).to_csv(ciel[2], index=False)
    print(f"chyby na valid ({a.system}): hazard {len(chyby['hazard'])}, product {len(chyby['product'])}")
    print(f"vzorka: {n_h} hazard + {n_p} product = {len(tab)} položiek")
    for c in ciel:
        print("uložené:", os.path.relpath(c, KOREN))


if __name__ == "__main__":
    main()
