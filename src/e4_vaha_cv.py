"""E4: výber váhy w pre spojenie klasifikátora a LLM (e4_llm.py --fuzia w) a poctivý odhad prínosu.

Spojenie: pre každý kandidát log p_klasifikátora + w · log p_LLM, vyberie sa najväčší (w = 0 je základ).
Ak sa w vyberie na celom valid a na ňom sa aj vyhodnotí, výsledok je optimistický. Preto opakovaná
dvojnásobná krížová validácia: valid sa náhodne rozdelí na polovice, na jednej sa vyberie w (najvyššie
ST2, pri zhode menšie w), na druhej sa zmeria ST2 oproti základu, potom naopak; opakuje sa s rôznymi
deleniami. Priemer rozdielov je odhad prínosu, ktorý by sme čakali na nových dátach.

Beží bez GPU, len z uložených odpovedí LLM (results/etapa2/llm_cache/<model>.jsonl).

Príklad:
  python src/e4_vaha_cv.py --model Qwen/Qwen2.5-3B-Instruct
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spolocne import SEED, VYSLEDKY, nacitaj, nacitaj_kandidatov, nacitaj_predikcie, oficialne_skore  # noqa: E402
import e4_llm  # noqa: E402

VARIANTY = {"oba": ("hazard", "product"), "product": ("product",), "hazard": ("hazard",)}


def nacitaj_z_cache(model, kandidati, split, k, max_znakov, prompt_fn=e4_llm.prompt_pismena):
    """Pre každý štítok: zoznamy kandidátov, log p_klasifikátora a log p_LLM (n × k)."""
    data = nacitaj(split)
    kand = nacitaj_kandidatov(kandidati, split).sort_values(["stitok", "id", "poradie"])
    cache = e4_llm.Cache(model)
    vystup = {}
    for lab in e4_llm.STITKY:
        T, PK, PL = [], [], []
        for i, g in kand[kand["stitok"] == lab].groupby("id"):
            trieda = g["trieda"].tolist()[:k]
            text = prompt_fn(data["title"][i], data["text"][i], lab, trieda, max_znakov)
            kluc = e4_llm.Cache.kluc(model, text)
            if kluc not in cache.data:
                sys.exit(f"v cache chýba odpoveď ({lab}, správa {i}); najprv spusti e4_llm.py na GPU")
            T.append(trieda)
            PK.append(e4_llm.pravdepodobnosti_klasifikatora(g["skore"].to_numpy()[:k]))
            PL.append(cache.data[kluc]["p"])
        vystup[lab] = (T, np.array(PK), np.array(PL))
    return data, vystup


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--kandidati", default="E12kens")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-znakov", type=int, default=1500)
    ap.add_argument("--vahy", type=float, nargs="+", default=[0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    ap.add_argument("--opakovania", type=int, default=50, help="počet náhodných delení valid na polovice")
    a = ap.parse_args()
    assert a.vahy[0] == 0, "prvá váha musí byť 0 (základ)"

    data, z = nacitaj_z_cache(a.model, a.kandidati, "valid", a.k, a.max_znakov)
    zaklad = nacitaj_predikcie(a.kandidati, "valid").sort_values("id").reset_index(drop=True)
    n = len(data)
    y_h, y_p = data["hazard"].to_numpy(), data["product"].to_numpy()

    # predikcie hazard a product pre každý štítok a váhu
    volby = {}
    for lab, (T, PK, PL) in z.items():
        for w in a.vahy:
            s = np.log(np.clip(PK, 1e-6, None)) + w * np.log(PL + 1e-4)
            j = s.argmax(axis=1)
            volby[lab, w] = np.array([T[i][j[i]] for i in range(n)], dtype=object)
    for lab in e4_llm.STITKY:  # w = 0 musí dať presne základný systém
        assert (volby[lab, 0] == zaklad[lab].to_numpy()).all(), f"{lab}: w = 0 sa nezhoduje so základom"

    def predikcie(variant, w):
        h = volby["hazard", w] if "hazard" in VARIANTY[variant] else volby["hazard", 0]
        p = volby["product", w] if "product" in VARIANTY[variant] else volby["product", 0]
        return h, p

    def st2(idx, variant, w):
        h, p = predikcie(variant, w)
        return oficialne_skore(y_h[idx], y_p[idx], h[idx], p[idx])

    vsetko = np.arange(n)
    print(f"ST2 na celom valid (n = {n}), w vybrané na tých istých dátach = optimistické:")
    tab = pd.DataFrame({variant: [st2(vsetko, variant, w) for w in a.vahy] for variant in VARIANTY},
                       index=pd.Index(a.vahy, name="w")).round(4)
    print(tab.to_string())

    rng = np.random.default_rng(SEED)
    delenia = [rng.permutation(n) for _ in range(a.opakovania)]
    riadky = []
    for variant in VARIANTY:
        zisky, vybrane = [], []
        for perm in delenia:
            A, B = perm[: n // 2], perm[n // 2:]
            for ladenie, test in ((A, B), (B, A)):
                sk = np.array([st2(ladenie, variant, w) for w in a.vahy])
                w_best = a.vahy[int(np.flatnonzero(sk >= sk.max() - 1e-12)[0])]
                vybrane.append(w_best)
                zisky.append(st2(test, variant, w_best) - st2(test, variant, 0))
        zisky = np.array(zisky)
        w_cele = a.vahy[int(np.flatnonzero(tab[variant].to_numpy() >= tab[variant].max() - 1e-12)[0])]
        riadky.append({
            "LLM pri": variant,
            "ST2 zisk (CV, priemer)": round(zisky.mean(), 4),
            "rozsah 2,5–97,5 %": f"[{np.percentile(zisky, 2.5):+.3f}; {np.percentile(zisky, 97.5):+.3f}]",
            "zisk > 0": f"{(zisky > 0).mean():.0%}",
            "najčastejšie w": pd.Series(vybrane).mode()[0],
            "w na celom valid": w_cele,
            "ST2 pri ňom": tab.loc[w_cele, variant],
        })
    vysl = pd.DataFrame(riadky)
    print(f"\nkrížová validácia: {a.opakovania} × 2 polovice (w sa vyberá na jednej, zisk sa meria na druhej)")
    print(vysl.to_string(index=False))
    priecinok = os.path.join(VYSLEDKY, "llm")
    os.makedirs(priecinok, exist_ok=True)
    meno = e4_llm.nazov_systemu(a.model, a.kandidati, 0, False)
    tab.to_csv(os.path.join(priecinok, f"vaha_krivka_{meno}_valid.csv"))
    vysl.to_csv(os.path.join(priecinok, f"vaha_cv_{meno}_valid.csv"), index=False)
    print(f"\nuložené: results/etapa2/llm/vaha_krivka_{meno}_valid.csv, vaha_cv_{meno}_valid.csv")


if __name__ == "__main__":
    main()
