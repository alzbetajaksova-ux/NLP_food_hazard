"""Zhoda dvoch anotátorov: Cohenova kappa, zhoda po kategóriách a zoznam nezhôd.

Predvolene porovná results/etapa2/chyby/anotacia_A.csv a anotacia_B.csv podľa stĺpca 'polozka'
a stĺpca 'kategoria'. Dá sa použiť aj na nové dáta: --a, --b, --kluc, --stlpce.

Beh: python src/zhoda.py
     python src/zhoda.py --a data/nove/anotacia_A.csv --b data/nove/anotacia_B.csv --kluc id \
                         --stlpce hazard-category product-category hazard product
"""
import argparse
import os
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spolocne import KOREN, VYSLEDKY  # noqa: E402

CHYBY = os.path.join(VYSLEDKY, "chyby")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default=os.path.join(CHYBY, "anotacia_A.csv"))
    ap.add_argument("--b", default=os.path.join(CHYBY, "anotacia_B.csv"))
    ap.add_argument("--kluc", default="polozka")
    ap.add_argument("--stlpce", nargs="+", default=["kategoria"])
    ap.add_argument("--nezhody", default=os.path.join(CHYBY, "nezhody.csv"))
    a = ap.parse_args()

    A = pd.read_csv(a.a, dtype=str, keep_default_na=False).set_index(a.kluc)
    B = pd.read_csv(a.b, dtype=str, keep_default_na=False).set_index(a.kluc)
    spolocne = A.index.intersection(B.index)
    vysledky, nezhody = [], []
    for st in a.stlpce:
        x, y = A.loc[spolocne, st].str.strip(), B.loc[spolocne, st].str.strip()
        vyplnene = (x != "") & (y != "")
        x, y = x[vyplnene], y[vyplnene]
        if len(x) == 0:
            print(f"{st}: zatiaľ nič vyplnené")
            continue
        vysledky.append({"stĺpec": st, "položiek": len(x), "zhoda": round((x == y).mean(), 3),
                         "Cohenova kappa": round(cohen_kappa_score(x, y), 3)})
        print(f"\n{st}: rozdelenie kategórií (A / B)")
        print(pd.DataFrame({"A": x.value_counts(), "B": y.value_counts()}).fillna(0).astype(int).to_string())
        print(f"\n{st}: matica zhody (riadky A, stĺpce B)")
        print(pd.crosstab(x, y).to_string())
        for k in x.index[x != y]:
            nezhody.append({a.kluc: k, "stĺpec": st, "A": x[k], "B": y[k], "rozhodnutie": "", "zdôvodnenie": ""})

    if vysledky:
        print("\nSúhrn:")
        print(pd.DataFrame(vysledky).to_string(index=False))
        pd.DataFrame(nezhody).to_csv(a.nezhody, index=False)
        print(f"nezhody ({len(nezhody)}) na rozhodnutie:", os.path.relpath(a.nezhody, KOREN))


if __name__ == "__main__":
    main()
