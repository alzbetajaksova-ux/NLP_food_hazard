"""E4: LLM vyberie hazard a produkt z top-k kandidátov iného systému (predvolene E12kens).

Postup pre každú správu a štítok (hazard, product):
  1. prompt = titulok + začiatok textu + očíslovaný zoznam k kandidátov,
  2. LLM odpovie jedným kandidátom (teplota 0),
  3. odpoveď sa porovná so zoznamom; ak nesedí, použije sa kandidát č. 1 (počet sa vypíše).
hazard-category sa odvodí z vybraného hazardu (hierarchia), product-category ostáva zo základu.

Každá odpoveď sa uloží do results/etapa2/llm_cache/<model>.jsonl (kľúč = hash modelu a promptu),
takže opakovaný beh nič neplatí a dá výsledok presne znova (reprodukovateľnosť bez prístupu k LLM).

Backendy:
  --backend skuska  vždy vyberie kandidáta č. 1, na overenie celej cesty bez LLM
  --backend openai  ľubovoľné API kompatibilné s OpenAI chat/completions (OpenAI, Groq, Ollama, vLLM …);
                    nastav premenné LLM_BASE_URL (napr. http://localhost:11434/v1) a LLM_API_KEY

Príklady:
  python src/e4_llm.py --kandidati E12kens --backend skuska
  python src/e4_llm.py --kandidati E12kens --backend openai --model <model> --limit 50   # skúška ceny a promptu
  python src/e4_llm.py --kandidati E12kens --backend openai --model <model>             # celý valid
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spolocne import (KOREN, VYSLEDKY, nacitaj, nacitaj_kandidatov, nacitaj_predikcie,  # noqa: E402
                      skore, uloz_predikcie)
from e3_hierarchia import mapa_hazard_kategoria  # noqa: E402

STITKY = ["hazard", "product"]
OTAZKA = {
    "hazard": "Which food hazard is the reason for this recall?",
    "product": "Which food product is being recalled?",
}
SYSTEMOVA = "You classify food recall reports. You always answer with exactly one option from the given list."


def prompt(titulok, text, stitok, kandidati, max_znakov):
    moznosti = "\n".join(f"{i}. {c}" for i, c in enumerate(kandidati, start=1))
    text = re.sub(r"\s+", " ", text)[:max_znakov]
    return (f"Recall report\nTitle: {titulok}\nText: {text}\n\n{OTAZKA[stitok]}\n"
            f"Options:\n{moznosti}\n\nAnswer with the option text only, copied exactly.")


def odpoved_na_kandidata(odpoved, kandidati):
    """Vráti vybraného kandidáta alebo None, ak odpoveď nezodpovedá žiadnej možnosti."""
    o = odpoved.strip().strip('"\'').strip()
    for varianta in (o, o.rstrip(".").strip()):
        for c in kandidati:
            if varianta.lower() == c.lower():
                return c
    cislo = re.fullmatch(r"(\d+)[.)]?(\s.*)?", o)
    if cislo and 1 <= int(cislo.group(1)) <= len(kandidati):
        return kandidati[int(cislo.group(1)) - 1]
    zhody = [c for c in kandidati if c.lower() in o.lower()]
    return max(zhody, key=len) if zhody else None


class Cache:
    def __init__(self, model):
        priecinok = os.path.join(VYSLEDKY, "llm_cache")
        os.makedirs(priecinok, exist_ok=True)
        self.cesta = os.path.join(priecinok, re.sub(r"[^\w.-]", "_", model) + ".jsonl")
        self.data = {}
        if os.path.exists(self.cesta):
            with open(self.cesta, encoding="utf-8") as f:
                for riadok in f:
                    z = json.loads(riadok)
                    self.data[z["kluc"]] = z["odpoved"]

    @staticmethod
    def kluc(model, text):
        return hashlib.sha256((model + "\n" + text).encode("utf-8")).hexdigest()

    def uloz(self, kluc, odpoved):
        self.data[kluc] = odpoved
        with open(self.cesta, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kluc": kluc, "odpoved": odpoved, "cas": time.strftime("%Y-%m-%d %H:%M:%S")},
                               ensure_ascii=False) + "\n")


def zavolaj_openai(model, text, pokusy=3):
    url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    telo = json.dumps({"model": model, "temperature": 0, "max_tokens": 40,
                       "messages": [{"role": "system", "content": SYSTEMOVA},
                                    {"role": "user", "content": text}]}).encode("utf-8")
    hlavicky = {"Content-Type": "application/json",
                "Authorization": "Bearer " + os.environ.get("LLM_API_KEY", "")}
    for pokus in range(pokusy):
        try:
            req = urllib.request.Request(url, data=telo, headers=hlavicky)
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            if pokus == pokusy - 1:
                raise
            print(f"  chyba ({e}), skúšam znova", flush=True)
            time.sleep(5 * (pokus + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kandidati", default="E12kens", help="systém, ktorého kandidátov a predikcie použijeme")
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--backend", choices=["skuska", "openai"], default="skuska")
    ap.add_argument("--model", default="skuska", help="názov modelu LLM (pri backende openai povinný)")
    ap.add_argument("--k", type=int, default=10, help="koľko kandidátov dostane LLM")
    ap.add_argument("--max-znakov", type=int, default=1500, help="koľko znakov textu správy ide do promptu")
    ap.add_argument("--limit", type=int, default=0, help="len prvých N správ (skúška), výsledok sa neuloží")
    ap.add_argument("--povolit-test", action="store_true", help="povolí beh na teste (až na konci)")
    a = ap.parse_args()
    if a.split == "test" and not a.povolit_test:
        sys.exit("Na test spúšťame až zmrazený systém na konci. Ak je to ten beh, pridaj --povolit-test.")

    data = nacitaj(a.split)
    zaklad = nacitaj_predikcie(a.kandidati, a.split).sort_values("id").reset_index(drop=True)
    kand = nacitaj_kandidatov(a.kandidati, a.split).sort_values(["stitok", "id", "poradie"])
    zoznamy = {lab: {i: g["trieda"].tolist()[:a.k] for i, g in kand[kand["stitok"] == lab].groupby("id")}
               for lab in STITKY}
    cache = Cache(a.model)
    n = a.limit or len(data)

    pred = zaklad.copy()
    statistika = {lab: {"volania": 0, "z_cache": 0, "nezhoda": 0} for lab in STITKY}
    t0 = time.time()
    for i in range(n):
        for lab in STITKY:
            kandidati = zoznamy[lab][i]
            text = prompt(data["title"][i], data["text"][i], lab, kandidati, a.max_znakov)
            kluc = Cache.kluc(a.model, text)
            if kluc in cache.data:
                odpoved = cache.data[kluc]
                statistika[lab]["z_cache"] += 1
            else:
                odpoved = kandidati[0] if a.backend == "skuska" else zavolaj_openai(a.model, text)
                statistika[lab]["volania"] += 1
                if a.backend != "skuska":
                    cache.uloz(kluc, odpoved)
            vyber = odpoved_na_kandidata(odpoved, kandidati)
            if vyber is None:
                statistika[lab]["nezhoda"] += 1
                vyber = kandidati[0]
            pred.loc[i, lab] = vyber
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} správ ({time.time() - t0:.0f} s)", flush=True)

    mapa = mapa_hazard_kategoria(nacitaj("train"))
    pred["hazard-category"] = pred["hazard"].map(mapa).fillna(pred["hazard-category"])
    print("štatistika:", json.dumps(statistika, ensure_ascii=False))

    if a.limit:
        vyrez = slice(0, n)
        s_llm = skore(data.iloc[vyrez], pred.iloc[vyrez])
        s_zak = skore(data.iloc[vyrez], zaklad.iloc[vyrez])
        print(f"skúška na prvých {n} správach (malá vzorka, len orientačne):")
        print(pd.DataFrame([{"systém": a.kandidati, **s_zak}, {"systém": "LLM", **s_llm}]).round(4).to_string(index=False))
        return

    kratky = re.sub(r"[^\w]", "", a.model.split("/")[-1])[:20]
    system = f"E4_{kratky}_{a.kandidati.removeprefix('E3_')}"
    cesta = uloz_predikcie(pred, system, a.split)
    print("uložené:", os.path.relpath(cesta, KOREN),
          f"| ďalej: python src/vyhodnot.py --split {a.split} {system} --porovnaj {a.kandidati}")


if __name__ == "__main__":
    main()
