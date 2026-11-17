"""Spoločné vyhodnotenie 2. etapy.

Príklady (z koreňa repozitára):
  python src/vyhodnot.py --split valid E0_svm E12kens
  python src/vyhodnot.py --split valid E12kens --porovnaj E0_svm --bootstrap 1000
  python src/vyhodnot.py --split valid --kandidati E0_svm

Systém sa zadáva menom (results/etapa2/predikcie/<meno>_<split>.csv) alebo cestou k CSV.
Test sa dá vyhodnotiť len s prepínačom --povolit-test: test spúšťame jediný raz na konci.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spolocne import (LABELY, SEED, VYSLEDKY, nacitaj, nacitaj_kandidatov,  # noqa: E402
                      nacitaj_predikcie, oficialne_skore, skore)


def nacitaj_system(meno, split):
    cesta = meno if meno.endswith(".csv") else os.path.join(VYSLEDKY, "predikcie", f"{meno}_{split}.csv")
    pred = nacitaj_predikcie(cesta)
    return pred.sort_values("id").reset_index(drop=True)


def bootstrap_skore(data, pred, B=1000, seed=SEED, ref=None):
    """Percentilový bootstrap ST1 a ST2. Ak je zadaný ref, počíta aj párový rozdiel pred − ref
    na tých istých vzorkách (rovnaké indexy pre oba systémy)."""
    rng = np.random.default_rng(seed)
    y = {lab: data[lab].values for lab in LABELY}
    p = {lab: pred[lab].values for lab in LABELY}
    r = {lab: ref[lab].values for lab in LABELY} if ref is not None else None
    out = {"ST1": [], "ST2": [], "ST1_rozdiel": [], "ST2_rozdiel": []}
    n = len(data)
    for _ in range(B):
        i = rng.integers(0, n, n)
        for st, (h, pr) in {"ST1": ("hazard-category", "product-category"), "ST2": ("hazard", "product")}.items():
            s = oficialne_skore(y[h][i], y[pr][i], p[h][i], p[pr][i])
            out[st].append(s)
            if r is not None:
                out[st + "_rozdiel"].append(s - oficialne_skore(y[h][i], y[pr][i], r[h][i], r[pr][i]))
    return {k: np.array(v) for k, v in out.items() if v}


def interval(a):
    return np.percentile(a, 2.5), np.percentile(a, 97.5)


def recall_at_k(data, kand, ks=(1, 3, 5, 10)):
    """Podiel správ, kde je správna trieda medzi prvými k kandidátmi, pre každý štítok."""
    riadky = []
    for lab in sorted(set(kand["stitok"]), key=LABELY.index):
        k_lab = kand[kand["stitok"] == lab]
        pravda = data[lab].reset_index(drop=True)
        trafene = k_lab[k_lab["trieda"].values == pravda.loc[k_lab["id"].values].values]
        najlepsie = trafene.groupby("id")["poradie"].min()
        riadok = {"štítok": lab}
        for k in ks:
            riadok[f"top-{k}"] = round(float((najlepsie <= k).sum()) / len(data), 4)
        riadky.append(riadok)
    return pd.DataFrame(riadky)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("systemy", nargs="*", help="mená systémov alebo cesty k CSV s predikciami")
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--porovnaj", help="referenčný systém pre párový bootstrap (napr. E0_svm)")
    ap.add_argument("--bootstrap", type=int, default=0, help="počet bootstrap výberov (0 = bez intervalov)")
    ap.add_argument("--kandidati", nargs="*", default=[], help="systémy, ktorým spočítať recall@k kandidátov")
    ap.add_argument("--povolit-test", action="store_true", help="povolí vyhodnotenie na teste")
    ap.add_argument("--uloz", help="uloží tabuľku skóre do CSV")
    a = ap.parse_args()

    if a.split == "test" and not a.povolit_test:
        sys.exit("Test vyhodnocujeme jediný raz na konci. Ak je to ten beh, pridaj --povolit-test.")

    data = nacitaj(a.split)
    ref = nacitaj_system(a.porovnaj, a.split) if a.porovnaj else None

    riadky = []
    for meno in a.systemy:
        pred = nacitaj_system(meno, a.split)
        assert len(pred) == len(data), f"{meno}: {len(pred)} riadkov, split má {len(data)}"
        riadok = {"systém": meno, **{k: round(v, 4) for k, v in skore(data, pred).items()}}
        if a.bootstrap:
            bs = bootstrap_skore(data, pred, a.bootstrap, ref=ref)
            for st in ["ST1", "ST2"]:
                lo, hi = interval(bs[st])
                riadok[f"{st} 95 % CI"] = f"[{lo:.3f}; {hi:.3f}]"
                if ref is not None:
                    lo, hi = interval(bs[st + "_rozdiel"])
                    rozdiel = riadok[st] - skore(data, ref)[st]
                    riadok[f"{st} − {a.porovnaj}"] = f"{rozdiel:+.3f} [{lo:+.3f}; {hi:+.3f}]"
                    riadok[f"{st} lepší v"] = f"{(bs[st + '_rozdiel'] > 0).mean():.0%}"
        riadky.append(riadok)

    if riadky:
        tab = pd.DataFrame(riadky)
        with pd.option_context("display.width", 200, "display.max_columns", 50):
            print(f"\nSplit: {a.split}  (n = {len(data)})")
            print(tab.to_string(index=False))
        if a.uloz:
            tab.to_csv(a.uloz, index=False)

    for meno in a.kandidati:
        cesta = meno if meno.endswith(".csv") else os.path.join(VYSLEDKY, "kandidati", f"{meno}_{a.split}.csv")
        print(f"\nRecall@k kandidátov: {meno} ({a.split})")
        print(recall_at_k(data, nacitaj_kandidatov(cesta)).to_string(index=False))


if __name__ == "__main__":
    main()
