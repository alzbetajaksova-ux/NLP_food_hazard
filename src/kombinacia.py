"""Pevná kombinácia dvoch systémov po štítkoch (pravidlo určené vopred, nie výberom pre každé semienko).

Príklad: hazard-category z transformera E2 a ostatné štítky z E0_svmvyber, pre každé semienko zvlášť,
aby sa dal uviesť priemer a rozptyl cez semienka:
  python src/kombinacia.py --zaklad E0_svmvyber --z E2_robertabase_s42 --stitky hazard-category --nazov E2k_s42
Pravidlo (ktoré štítky z ktorého systému) sa určí podľa priemeru cez semienka na valid.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spolocne import (nacitaj, nacitaj_kandidatov, nacitaj_predikcie, skore,  # noqa: E402
                      uloz_kandidatov, uloz_predikcie)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zaklad", required=True)
    ap.add_argument("--z", required=True, help="systém, z ktorého sa berú zvolené štítky")
    ap.add_argument("--stitky", nargs="+", required=True)
    ap.add_argument("--nazov", required=True)
    a = ap.parse_args()
    for split in ["valid", "test"]:
        p = nacitaj_predikcie(a.zaklad, split).sort_values("id").reset_index(drop=True)
        q = nacitaj_predikcie(a.z, split).sort_values("id").reset_index(drop=True)
        for lab in a.stitky:
            p[lab] = q[lab].values
        k1, k2 = nacitaj_kandidatov(a.zaklad, split), nacitaj_kandidatov(a.z, split)
        k = pd.concat([k1[~k1["stitok"].isin(a.stitky)], k2[k2["stitok"].isin(a.stitky)]])
        uloz_predikcie(p, a.nazov, split)
        uloz_kandidatov(k, a.nazov, split)
    s = skore(nacitaj("valid"), nacitaj_predikcie(a.nazov, "valid"))
    print(f"{a.nazov} na valid: ST1 {s['ST1']:.4f}, ST2 {s['ST2']:.4f}")


if __name__ == "__main__":
    main()
