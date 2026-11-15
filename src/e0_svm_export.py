"""E0: baseline TF-IDF + LinearSVC z 1. etapy (title + text, class_weight='balanced').

Uloží pre valid aj test:
  results/etapa2/predikcie/E0_svm_<split>.csv   predikcie všetkých štyroch štítkov
  results/etapa2/kandidati/E0_svm_<split>.csv   top-10 tried so skóre (decision_function)

Nastavenia sú rovnaké ako vo víťaznom variante baseline, takže skóre na teste sa musí
zhodovať s results/summary.csv (ST1 0,7225, ST2 0,4097). Skript to na konci overí.
Beh: python src/e0_svm_export.py   (asi pol minúty na CPU)

Variant --titulok (systém E0_svmtit): k príznakom title + text sa pridajú znakové n-gramy (3 až 5)
samotného titulku, takže model kladie väčší dôraz na titulok, kde je zvyčajne názov produktu.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spolocne import (KOREN, LABELY, SEED, TOP_K, nacitaj, skore, tokenizuj,  # noqa: E402
                      uloz_kandidatov, uloz_predikcie, vstup)

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--titulok", action="store_true", help="pridať znakové n-gramy titulku (E0_svmtit)")
    a = ap.parse_args()
    SYSTEM = "E0_svmtit" if a.titulok else "E0_svm"
    np.random.seed(SEED)
    train, valid, test = nacitaj("train"), nacitaj("valid"), nacitaj("test")
    vec = TfidfVectorizer(tokenizer=tokenizuj, token_pattern=None, lowercase=False,
                          sublinear_tf=True, min_df=2)
    X = {"train": vec.fit_transform(vstup(train)),      # slovník a IDF iba z train
         "valid": vec.transform(vstup(valid)), "test": vec.transform(vstup(test))}
    if a.titulok:
        vec_t = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
        X_t = {"train": vec_t.fit_transform(train["title"]),
               "valid": vec_t.transform(valid["title"]), "test": vec_t.transform(test["title"])}
        X = {s: hstack([X[s], X_t[s]]).tocsr() for s in X}
    data = {"valid": valid, "test": test}

    pred = {s: pd.DataFrame({"id": np.arange(len(data[s]))}) for s in data}
    kand = {s: [] for s in data}
    t = time.time()
    for lab in LABELY:
        model = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED).fit(X["train"], train[lab])
        for s in data:
            d = model.decision_function(X[s])
            poradie = np.argsort(-d, axis=1)[:, :TOP_K]
            pred[s][lab] = model.classes_[poradie[:, 0]]
            n = len(data[s])
            kand[s].append(pd.DataFrame({
                "id": np.repeat(np.arange(n), TOP_K),
                "stitok": lab,
                "poradie": np.tile(np.arange(1, TOP_K + 1), n),
                "trieda": model.classes_[poradie].ravel(),
                "skore": np.round(np.take_along_axis(d, poradie, axis=1).ravel(), 5),
            }))
        print(f"{lab}: hotovo ({time.time() - t:.0f} s)")

    for s in data:
        print("uložené:", os.path.relpath(uloz_predikcie(pred[s], SYSTEM, s), KOREN),
              "|", os.path.relpath(uloz_kandidatov(pd.concat(kand[s]), SYSTEM, s), KOREN))

    if a.titulok:
        return
    # kontrola zhody s 1. etapou
    t1 = skore(test, pred["test"])
    ref = pd.read_csv(os.path.join(KOREN, "results", "summary.csv")).set_index("model")
    ref = ref.loc["SVM (title+text, balanced)"]
    for st in ["ST1", "ST2"]:
        assert abs(t1[st] - ref[st]) < 1e-4, f"{st}: {t1[st]:.4f} sa nezhoduje so summary.csv ({ref[st]})"
    print(f"kontrola: test ST1 {t1['ST1']:.4f}, ST2 {t1['ST2']:.4f} = summary.csv z 1. etapy")


if __name__ == "__main__":
    main()
