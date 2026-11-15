"""Pre každý štítok vyberie na valid systém s najvyšším macro-F1 a poskladá z nich nový systém.

Príklad: python src/vyber_po_stitkoch.py --systemy E0_svm E0_svmtit --nazov E0_svmvyber
Výber sa robí len podľa valid; rovnaká voľba sa potom použije na test (predikcie aj kandidáti).
Hodí sa aj neskôr, napr. na kombináciu transformera a SVM (každý štítok z lepšieho z nich).
"""
import argparse
import json
import os
import sys

import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spolocne import (KOREN, LABELY, VYSLEDKY, nacitaj, nacitaj_kandidatov,  # noqa: E402
                      nacitaj_predikcie, skore, uloz_kandidatov, uloz_predikcie)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systemy", nargs="+", required=True)
    ap.add_argument("--nazov", required=True)
    a = ap.parse_args()

    valid = nacitaj("valid")
    pv = {s: nacitaj_predikcie(s, "valid").sort_values("id").reset_index(drop=True) for s in a.systemy}
    tabulka, volba = [], {}
    for lab in LABELY:
        f1 = {s: f1_score(valid[lab], pv[s][lab], average="macro", zero_division=0) for s in a.systemy}
        volba[lab] = max(f1, key=f1.get)
        tabulka.append({"štítok": lab, **{s: round(v, 4) for s, v in f1.items()}, "vybraný": volba[lab]})
    print("macro-F1 na valid po štítkoch:")
    print(pd.DataFrame(tabulka).to_string(index=False))

    for split in ["valid", "test"]:
        pred = {s: nacitaj_predikcie(s, split).sort_values("id").reset_index(drop=True) for s in set(volba.values())}
        kand = {s: nacitaj_kandidatov(s, split) for s in set(volba.values())}
        nove = pd.DataFrame({"id": pred[volba[LABELY[0]]]["id"]})
        for lab in LABELY:
            nove[lab] = pred[volba[lab]][lab].values
        k = pd.concat([kand[volba[lab]][kand[volba[lab]]["stitok"] == lab] for lab in LABELY])
        uloz_predikcie(nove, a.nazov, split)
        uloz_kandidatov(k, a.nazov, split)
    os.makedirs(os.path.join(VYSLEDKY, "logy"), exist_ok=True)
    with open(os.path.join(VYSLEDKY, "logy", f"{a.nazov}.json"), "w", encoding="utf-8") as f:
        json.dump({"system": a.nazov, "volba_podla_valid": volba, "f1_valid": tabulka}, f, ensure_ascii=False, indent=2)
    s = skore(valid, nacitaj_predikcie(a.nazov, "valid"))
    print(f"{a.nazov} na valid: ST1 {s['ST1']:.4f}, ST2 {s['ST2']:.4f}  (uložené valid aj test)")


if __name__ == "__main__":
    main()
