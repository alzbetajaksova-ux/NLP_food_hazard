"""Zostaví všetky systémy 2. etapy, ktoré nepotrebujú GPU, z uložených výstupov.

Poradie krokov (všetko na CPU, spolu asi 2 minúty):
  1. E0_svm      baseline SVM z 1. etapy (predikcie a top-10 kandidátov)
  2. E0_svmtit   SVM so znakovými n-gramami titulku (medzikrok)
  3. E0_svmvyber pre každý štítok lepší z E0_svm a E0_svmtit podľa valid
  4. E2ens, E1bens ensemble troch semienok transformera (vyžaduje uložené behy E2 a E1b z Colabu)
  5. E2kens      hazard-category z E2ens, ostatné z E0_svmvyber
  6. E12kens     E2kens + hazard a product z E1bens (zatiaľ najlepší systém)
S prepínačom --ablacie navyše E3 (hierarchia a produkt z titulku) nad E0_svm a nad E12kens.

V gite sú len výstupy transformera (nedajú sa rýchlo zopakovať) a výsledné systémy E0_svm, E0_svmvyber,
E2kens a E12kens; medzikroky a ablácie vznikajú týmto skriptom a sú v .gitignore.
Beh: python src/zostav_systemy.py [--ablacie]
"""
import argparse
import os
import subprocess
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
SEMIENKA = [42, 43, 44]


def spusti(*argumenty):
    prikaz = [sys.executable, os.path.join(SRC, argumenty[0]), *argumenty[1:]]
    print(">>>", " ".join(prikaz[1:]), flush=True)
    subprocess.run(prikaz, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablacie", action="store_true", help="vytvoriť aj E3 nad E0_svm a nad E12kens")
    a = ap.parse_args()

    spusti("e0_svm_export.py")
    spusti("e0_svm_export.py", "--titulok")
    spusti("vyber_po_stitkoch.py", "--systemy", "E0_svm", "E0_svmtit", "--nazov", "E0_svmvyber")
    spusti("ensemble.py", "--systemy", *[f"E2_robertabase_s{k}" for k in SEMIENKA], "--nazov", "E2ens")
    spusti("ensemble.py", "--systemy", *[f"E1b_robertabase_s{k}" for k in SEMIENKA], "--nazov", "E1bens")
    spusti("kombinacia.py", "--zaklad", "E0_svmvyber", "--z", "E2ens", "--stitky", "hazard-category",
           "--nazov", "E2kens")
    spusti("kombinacia.py", "--zaklad", "E2kens", "--z", "E1bens", "--stitky", "hazard", "product",
           "--nazov", "E12kens")
    if a.ablacie:
        spusti("e3_hierarchia.py", "--zaklad", "E0_svm")
        spusti("e3_hierarchia.py", "--zaklad", "E12kens")


if __name__ == "__main__":
    main()
