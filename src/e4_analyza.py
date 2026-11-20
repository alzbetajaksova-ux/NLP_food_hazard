"""E4: kde LLM pomáha a kde škodí (analýza zmien oproti základnému systému, bez GPU).

Číta detailné výstupy behov e4_llm.py (results/etapa2/llm/detail_<systém>_<split>.csv), v ktorých je pre
každú správu a štítok kandidát č. 1 základného systému (E12kens), výber LLM a výsledná odpoveď.
Zmena odpovede oproti kandidátovi č. 1 je:
  pomohol   kandidát 1 bol zlý, výsledok je správny
  uškodil   kandidát 1 bol správny, výsledok je zlý
  zlé → zlé kandidát 1 bol zlý a výsledok je iná zlá trieda

Tabuľky (všetky na valid, pre hazard a product zvlášť):
  1 prehľad zmien pre každý systém
  2 podľa poradia správnej triedy medzi kandidátmi klasifikátora
  3 podľa istoty klasifikátora (pravdepodobnosť kandidáta 1 normalizovaná na top-k)
  4 podľa početnosti správnej triedy v train
  5 podľa kategórie (hazard-category, product-category) správnej triedy
  6 vplyv na macro-F1 po triedach (prvý systém v zozname), ako v oficiálnom skóre
  7 najčastejšie zámeny (kandidát 1 → výsledok) pri pomohol a uškodil
  8 príklady správ (prvý systém)
Výstup: results/etapa2/llm/analyza/*.csv a súhrn analyza_E4_<split>.md.

Príklad:
  python src/e4_analyza.py
  python src/e4_analyza.py --systemy E4_Qwen253BInstruct_E12kens_f030_prod E4_Qwen253BInstruct_E12kens
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spolocne import SEED, VYSLEDKY, nacitaj, nacitaj_kandidatov  # noqa: E402
import e4_llm  # noqa: E402

SYSTEMY = [
    ("E4_Qwen253BInstruct_E12kens_f030_prod", "3B spojenie, product (hlavný)"),
    ("E4_Qwen253BInstruct_E12kens_f010", "3B spojenie, oba"),
    ("E4_Qwen253BInstruct_E12kens", "3B sám"),
    ("E4_Qwen257BInstructbnb4_E12kens_f020_prod", "7B spojenie, product"),
    ("E4_Qwen257BInstructbnb4_E12kens", "7B sám"),
]
KATEGORIA = {"hazard": "hazard-category", "product": "product-category"}
PRIECINOK = os.path.join(VYSLEDKY, "llm", "analyza")


def nacitaj_detail(system, split, data, pocty_train, p_klas):
    d = pd.read_csv(os.path.join(VYSLEDKY, "llm", f"detail_{system}_{split}.csv"))
    zmena = d["vysledok"] != d["kandidat_1"]
    d["pomohol"] = zmena & (d["vysledok"] == d["spravna"])
    d["uskodil"] = zmena & (d["kandidat_1"] == d["spravna"])
    d["zle_zle"] = zmena & ~d["pomohol"] & ~d["uskodil"]
    d["zmena"] = zmena
    d["kategoria"] = [data[KATEGORIA[lab]][i] for i, lab in zip(d["id"], d["stitok"])]
    d["pocet_v_train"] = [pocty_train[lab].get(t, 0) for t, lab in zip(d["spravna"], d["stitok"])]
    d["p_klas_1"] = [p_klas[lab][i] for i, lab in zip(d["id"], d["stitok"])]
    return d


def kos_poradie(r):
    return "mimo top-10" if r == 0 else {1: "1", 2: "2"}.get(r, "3–5" if r <= 5 else "6–10")


def kos_istota(p):
    for hranica, meno in ((0.3, "< 0,3"), (0.5, "0,3–0,5"), (0.7, "0,5–0,7"), (0.9, "0,7–0,9")):
        if p < hranica:
            return meno
    return "≥ 0,9"


def kos_pocet(c):
    if c == 0:
        return "0 (nie je v train)"
    for hranica, meno in ((5, "1–5"), (20, "6–20"), (100, "21–100")):
        if c <= hranica:
            return meno
    return "> 100"


PORADIA = {
    "poradie správnej triedy": (kos_poradie, ["1", "2", "3–5", "6–10", "mimo top-10"]),
    "istota klasifikátora (p kandidáta 1)": (kos_istota, ["< 0,3", "0,3–0,5", "0,5–0,7", "0,7–0,9", "≥ 0,9"]),
    "počet správ s triedou v train": (kos_pocet, ["0 (nie je v train)", "1–5", "6–20", "21–100", "> 100"]),
}


def podla_kosov(detaily, lab, stlpec, fn, poradie, mena, popisy):
    """Riadky = koše, stĺpce: n, presnosť základu a pre každý systém '+pomohol / −uškodil'."""
    prvy = detaily[mena[0]]
    prvy = prvy[prvy["stitok"] == lab]
    kos = prvy[stlpec].map(fn) if fn else prvy[stlpec]
    tab = pd.DataFrame({"n": kos.value_counts(),
                        "presnosť základu": (prvy["kandidat_1"] == prvy["spravna"]).groupby(kos).mean()})
    for m in mena:
        d = detaily[m][detaily[m]["stitok"] == lab]
        if not d["zmena"].any():
            continue  # LLM sa pri tomto štítku nepoužil
        k = d[stlpec].map(fn) if fn else d[stlpec]
        g = d.groupby(k)
        tab[popisy.get(m, m)] = [f"+{g['pomohol'].sum().get(x, 0)} / −{g['uskodil'].sum().get(x, 0)}"
                                 for x in tab.index]
    if poradie:
        tab = tab.reindex([x for x in poradie if x in tab.index])
    else:
        tab = tab.sort_values("n", ascending=False)
    tab["presnosť základu"] = (tab["presnosť základu"] * 100).round(1)
    return tab


def f1_po_triedach(data, detail):
    """Zmena F1 každej triedy tak, ako vstupuje do ST2: hazard na všetkých správach, product len na
    správach so správnym hazardom (hazard výsledného systému). Vráti tabuľku tried a prehľad."""
    n = len(data)
    pred = {lab: detail[detail["stitok"] == lab].sort_values("id") for lab in e4_llm.STITKY}
    riadky, prehlad = [], []
    for lab in e4_llm.STITKY:
        y = data[lab].to_numpy()
        zak = pred[lab]["kandidat_1"].to_numpy()
        e4 = pred[lab]["vysledok"].to_numpy()
        if lab == "product":
            h = data["hazard"].to_numpy()
            ok_zak = pred["hazard"]["kandidat_1"].to_numpy() == h
            ok_e4 = pred["hazard"]["vysledok"].to_numpy() == h
        else:
            ok_zak = ok_e4 = np.ones(n, bool)
        vysl = {}
        for meno, p, ok in (("základ", zak, ok_zak), ("E4", e4, ok_e4)):
            triedy = np.union1d(y[ok], p[ok])  # sklearn macro-F1 berie zjednotenie skutočných a predikovaných
            f1 = f1_score(y[ok], p[ok], labels=triedy, average=None, zero_division=0)
            vysl[meno] = (dict(zip(triedy, f1)), len(triedy), f1.mean(), set(triedy) - set(y[ok]))
        prehlad.append({
            "štítok": lab,
            "macro-F1 základ": round(vysl["základ"][2], 4), "macro-F1 E4": round(vysl["E4"][2], 4),
            "tried v priemere: základ": vysl["základ"][1], "tried v priemere: E4": vysl["E4"][1],
            "predikované triedy, ktoré v (hodnotenej časti) valid nie sú: základ": len(vysl["základ"][3]),
            "…: E4": len(vysl["E4"][3]),
        })
        m = vysl["E4"][1]
        for t in sorted(set(vysl["základ"][0]) | set(vysl["E4"][0])):
            fz, fe = vysl["základ"][0].get(t, 0.0), vysl["E4"][0].get(t, 0.0)
            if fz != fe or (t in vysl["E4"][0]) != (t in vysl["základ"][0]):
                riadky.append({"štítok": lab, "trieda": t, "správ v valid": int((y == t).sum()),
                               "F1 základ": round(fz, 3), "F1 E4": round(fe, 3), "zmena F1": round(fe - fz, 3),
                               "príspevok k macro-F1 (≈ zmena / počet tried)": round((fe - fz) / m, 4),
                               "len predikovaná (nie je v valid)": t in vysl["E4"][3] or t in vysl["základ"][3]})
    return pd.DataFrame(riadky).sort_values(["štítok", "zmena F1"]), pd.DataFrame(prehlad)


def zameny(d, lab, stlpec, pocet=10):
    z = d[(d["stitok"] == lab) & d[stlpec]]
    return (z.groupby(["kandidat_1", "vysledok"]).size().rename("počet").reset_index()
            .sort_values("počet", ascending=False).head(pocet)
            .rename(columns={"kandidat_1": "kandidát 1 (základ)", "vysledok": "výsledok E4"}))


def md_tabulka(df, index=True):
    df = df.reset_index() if index else df
    hlavicka = "| " + " | ".join(map(str, df.columns)) + " |"
    oddel = "| " + " | ".join("---" for _ in df.columns) + " |"
    riadky = ["| " + " | ".join(str(v).replace("|", "/") for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([hlavicka, oddel, *riadky])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systemy", nargs="+", default=[s for s, _ in SYSTEMY],
                    help="systémy E4 (prvý je hlavný: tabuľky 6 až 8 sú preň)")
    ap.add_argument("--kandidati", default="E12kens")
    ap.add_argument("--split", default="valid", choices=["valid"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--priklady", type=int, default=6)
    a = ap.parse_args()

    data, train = nacitaj(a.split), nacitaj("train")
    pocty_train = {lab: train[lab].value_counts().to_dict() for lab in e4_llm.STITKY}
    kand = nacitaj_kandidatov(a.kandidati, a.split).sort_values(["stitok", "id", "poradie"])
    p_klas = {lab: {i: e4_llm.pravdepodobnosti_klasifikatora(g["skore"].to_numpy()[:a.k])[0]
                    for i, g in kand[kand["stitok"] == lab].groupby("id")} for lab in e4_llm.STITKY}
    popisy = dict(SYSTEMY)
    detaily = {s: nacitaj_detail(s, a.split, data, pocty_train, p_klas) for s in a.systemy}
    for s, d in detaily.items():  # kandidát 1 musí byť vo všetkých behoch ten istý (ten istý základ)
        assert (d["kandidat_1"].to_numpy() == detaily[a.systemy[0]]["kandidat_1"].to_numpy()).all(), s
    os.makedirs(PRIECINOK, exist_ok=True)
    md = [f"# Analýza E4 na {a.split}: kde LLM pomáha a kde škodí", "",
          f"Základ: {a.kandidati} (kandidát č. 1). Vytvorené skriptom `src/e4_analyza.py`; zmena = výsledok E4 "
          "iný než kandidát 1. Zápis „+a / −b“: a správ opravil (pomohol), b pokazil (uškodil).", "",
          "Systémy: " + "; ".join(f"`{s}` = {popisy.get(s, s)}" for s in a.systemy), ""]

    def uloz(df, meno, nadpis, index=True, poznamka=None):
        df.to_csv(os.path.join(PRIECINOK, f"{meno}_{a.split}.csv"), index=index)
        md.extend([f"## {nadpis}", ""] + ([poznamka, ""] if poznamka else []) + [md_tabulka(df, index), ""])
        print(f"\n=== {nadpis} ===")
        print(df.to_string(index=index))

    # 1 prehľad
    riadky = []
    for s, d in detaily.items():
        for lab in e4_llm.STITKY:
            x = d[d["stitok"] == lab]
            riadky.append({"systém": popisy.get(s, s), "štítok": lab, "zmenené": int(x["zmena"].sum()),
                           "pomohol": int(x["pomohol"].sum()), "uškodil": int(x["uskodil"].sum()),
                           "zlé → zlé": int(x["zle_zle"].sum()),
                           "čistý zisk": int(x["pomohol"].sum() - x["uskodil"].sum()),
                           "presnosť základ %": round((x["kandidat_1"] == x["spravna"]).mean() * 100, 1),
                           "presnosť E4 %": round((x["vysledok"] == x["spravna"]).mean() * 100, 1)})
    uloz(pd.DataFrame(riadky), "1_prehlad", "1 Prehľad zmien", index=False)

    # 2 až 5 podľa košov
    for c, (nadpis, stlpec, fn, poradie) in enumerate([
            ("poradie správnej triedy", "poradie_spravnej", *PORADIA["poradie správnej triedy"]),
            ("istota klasifikátora (p kandidáta 1)", "p_klas_1", *PORADIA["istota klasifikátora (p kandidáta 1)"]),
            ("počet správ s triedou v train", "pocet_v_train", *PORADIA["počet správ s triedou v train"]),
            ("kategória správnej triedy", "kategoria", None, None)], start=2):
        for lab in e4_llm.STITKY:
            tab = podla_kosov(detaily, lab, stlpec, fn, poradie, a.systemy, popisy)
            tab.index.name = nadpis
            uloz(tab, f"{c}_{stlpec}_{lab}", f"{c} {lab}: podľa {nadpis}")

    # 6 macro-F1 po triedach (hlavný systém)
    hlavny = a.systemy[0]
    triedy, prehlad = f1_po_triedach(data, detaily[hlavny])
    uloz(prehlad, "6_macro_f1_prehlad", f"6 Macro-F1 ako v ST2: {popisy.get(hlavny, hlavny)}", index=False,
         poznamka="Product sa hodnotí len na správach so správnym hazardom. Macro-F1 sa priemeruje cez zjednotenie "
                  "skutočných a predikovaných tried, takže predikovaná trieda, ktorá v hodnotenej časti nie je, "
                  "pridá do priemeru nulu.")
    for lab in e4_llm.STITKY:
        t = triedy[triedy["štítok"] == lab].drop(columns="štítok")
        if len(t):
            vyber = pd.concat([t.head(8), t.tail(8)]).drop_duplicates()
            uloz(vyber, f"6_f1_triedy_{lab}", f"6 {lab}: triedy s najväčšou stratou a ziskom F1", index=False)
            t.to_csv(os.path.join(PRIECINOK, f"6_f1_triedy_{lab}_vsetky_{a.split}.csv"), index=False)

    # 7 zámeny
    for s in a.systemy[:1] + [x for x in a.systemy if "_f" not in x.split("E12kens")[-1]][:1]:
        for lab in e4_llm.STITKY:
            for stlpec, slovo in (("uskodil", "uškodil"), ("pomohol", "pomohol")):
                z = zameny(detaily[s], lab, stlpec)
                if len(z):
                    uloz(z, f"7_zameny_{stlpec}_{lab}_{s}", f"7 {popisy.get(s, s)}, {lab}: najčastejšie zámeny, keď {slovo}",
                         index=False)

    # 8 príklady
    d = detaily[hlavny]
    rng = np.random.default_rng(SEED)
    riadky = []
    for lab in e4_llm.STITKY:
        for stlpec, slovo in (("pomohol", "pomohol"), ("uskodil", "uškodil")):
            x = d[(d["stitok"] == lab) & d[stlpec]]
            for _, r in x.iloc[rng.permutation(len(x))[:a.priklady]].sort_values("id").iterrows():
                riadky.append({"štítok": lab, "LLM": slovo, "id": r["id"], "titulok": data["title"][r["id"]][:110],
                               "správna": r["spravna"], "kandidát 1": r["kandidat_1"], "E4": r["vysledok"],
                               "poradie správnej": r["poradie_spravnej"], "p klasifikátora (kand. 1)": round(r["p_klas_1"], 3),
                               "p LLM (jeho výber)": round(r["p_llm"], 3)})
    if riadky:
        uloz(pd.DataFrame(riadky), "8_priklady", f"8 Príklady: {popisy.get(hlavny, hlavny)} (náhodne, seed {SEED})", index=False)

    with open(os.path.join(PRIECINOK, f"analyza_E4_{a.split}.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md))
    print(f"\nuložené: results/etapa2/llm/analyza/ (csv a analyza_E4_{a.split}.md)")


if __name__ == "__main__":
    main()
