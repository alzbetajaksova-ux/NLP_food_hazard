"""Ensemble viacerých behov (napr. 3 semienka transformera): pravdepodobnosti z kandidátov sa sčítajú.

Pri štítkoch s najviac 10 triedami (hazard-category) sú v kandidátoch celé rozdelenia, takže ide o presný
priemer pravdepodobností; pri ostatných je to súčet cez top-10 (chýbajúca trieda = 0), teda približný.
Príklad: python src/ensemble.py --systemy E2_robertabase_s42 E2_robertabase_s43 E2_robertabase_s44 --nazov E2ens
Nepotrebuje GPU: pracuje len s uloženými kandidátmi.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spolocne import (LABELY, TOP_K, nacitaj, nacitaj_kandidatov, skore,  # noqa: E402
                      uloz_kandidatov, uloz_predikcie, nacitaj_predikcie)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systemy", nargs="+", required=True)
    ap.add_argument("--nazov", required=True)
    a = ap.parse_args()
    for split in ["valid", "test"]:
        n = len(nacitaj(split))
        k = pd.concat([nacitaj_kandidatov(s, split) for s in a.systemy])
        g = k.groupby(["stitok", "id", "trieda"], as_index=False)["skore"].sum()
        g["skore"] = (g["skore"] / len(a.systemy)).round(5)
        g = g.sort_values(["stitok", "id", "skore"], ascending=[True, True, False])
        g["poradie"] = g.groupby(["stitok", "id"]).cumcount() + 1
        g = g[g["poradie"] <= TOP_K]
        pred = pd.DataFrame({"id": np.arange(n)})
        for lab in LABELY:
            prvy = g[(g["stitok"] == lab) & (g["poradie"] == 1)].set_index("id")["trieda"]
            pred[lab] = prvy.reindex(range(n)).values
        uloz_predikcie(pred, a.nazov, split)
        uloz_kandidatov(g, a.nazov, split)
    s = skore(nacitaj("valid"), nacitaj_predikcie(a.nazov, "valid"))
    print(f"{a.nazov} na valid: " + ", ".join(f"{x} {v:.4f}" for x, v in s.items()))


if __name__ == "__main__":
    main()
