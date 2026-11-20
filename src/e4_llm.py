"""E4: LLM vyberie hazard a produkt z top-k kandidátov iného systému (predvolene E12kens).

Postup pre každú správu a štítok (hazard, product):
  1. prompt = titulok + začiatok textu + zoznam k kandidátov v poradí systému,
  2. LLM vyberie jedného kandidáta (deterministicky, bez náhody),
  3. výsledný štítok je výber LLM; s --prah iba vtedy, keď si je LLM aspoň takto istý,
     inak ostane kandidát č. 1 zo základného systému; s --fuzia w sa spojí istota klasifikátora
     a LLM: argmax log p_klasifikátora + w · log p_LLM (w = 0 je základ, väčšie w = viac LLM).
     Váha sa vyberá krížovou validáciou na valid: src/e4_vaha_cv.py.
     --stitky-llm product použije LLM len pri produkte (hazard ostane zo základu).
hazard-category a product-category ostávajú zo základného systému. Pri E12kens je hazard-category
z transformera lepšia než odvodená z hazardu (zápisník, E3 nad E12kens: ST1 0,802 → 0,788);
odvodenie z vybraného hazardu sa dá zapnúť prepínačom --hierarchia (ablácia).

Backendy:
  --backend skuska  vždy vyberie kandidáta č. 1, na overenie celej cesty bez LLM a bez GPU
  --backend hf      lokálny model cez knižnicu transformers na GPU (datalab). Kandidáti sú označení
                    písmenami A, B, C …; LLM nič negeneruje, berie sa pravdepodobnosť, že by ako prvé
                    napísal dané písmeno (jeden prechod modelom na otázku). Výsledkom je rozdelenie
                    pravdepodobností cez kandidátov, odpoveď je vždy platná a dá sa nastaviť prah istoty.
                    Písmená namiesto čísel, lebo tokenizér Qwen delí čísla na číslice ("10" = "1" + "0").
  --backend openai  API kompatibilné s OpenAI chat/completions (OpenAI, Groq, Ollama, vLLM …);
                    LLM napíše odpoveď textom a tá sa porovná so zoznamom; ak nesedí, použije sa
                    kandidát č. 1 (počet sa vypíše). Nastav LLM_BASE_URL a LLM_API_KEY.

Každá odpoveď sa hneď uloží do results/etapa2/llm_cache/<model>.jsonl (kľúč = hash modelu a promptu):
prerušený beh pokračuje, kde skončil, opakovaný beh (napr. iný --prah) nepotrebuje GPU a dá presne
ten istý výsledok, takže sa dá reprodukovať aj bez prístupu k LLM.

Príklady:
  python src/e4_llm.py --backend skuska
  python src/e4_llm.py --backend hf --model Qwen/Qwen2.5-3B-Instruct --model-cesta <priečinok> --limit 50
  python src/e4_llm.py --backend hf --model Qwen/Qwen2.5-3B-Instruct --model-cesta <priečinok>
  python src/e4_llm.py --backend hf --model Qwen/Qwen2.5-3B-Instruct --prah 0.7   # z cache, bez GPU
  python src/e4_llm.py --backend hf --model Qwen/Qwen2.5-3B-Instruct --fuzia 0.3 --stitky-llm product
"""
import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
import urllib.request

import numpy as np
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
PISMENA = "ABCDEFGHIJKLMNOPQRST"

# backend openai: odpoveď textom
SYSTEMOVA = "You classify food recall reports. You always answer with exactly one option from the given list."
# backend hf: odpoveď písmenom
SYSTEMOVA_HF = ("You classify food recall reports. You always answer with the letter of exactly one "
                "option from the given list.")


def _text(text, max_znakov):
    return re.sub(r"\s+", " ", text)[:max_znakov]


def prompt(titulok, text, stitok, kandidati, max_znakov):
    """Prompt pre backend openai (možnosti očíslované, odpoveď textom)."""
    moznosti = "\n".join(f"{i}. {c}" for i, c in enumerate(kandidati, start=1))
    return (f"Recall report\nTitle: {titulok}\nText: {_text(text, max_znakov)}\n\n{OTAZKA[stitok]}\n"
            f"Options:\n{moznosti}\n\nAnswer with the option text only, copied exactly.")


def prompt_pismena(titulok, text, stitok, kandidati, max_znakov, priklady=None, poradie=False):
    """Prompt pre backend hf (možnosti označené písmenami, odpoveď jedným písmenom).
    priklady: trieda -> (počet správ v train, príklad titulku), pridá sa ku každej možnosti (prompt v2, v3);
    poradie: povie modelu, že možnosti zoradil klasifikátor a A je jeho najlepší tip (prompt v3)."""
    riadky = []
    for i, c in enumerate(kandidati):
        r = f"{PISMENA[i]}. {c}"
        if priklady is not None and c in priklady:
            pocet, titulok_pr = priklady[c]
            r += f' ({pocet} training reports, e.g. "{titulok_pr}")'
        riadky.append(r)
    uvod = (UVOD_PRIKLADY + "\n" if priklady is not None else "") + (UVOD_PORADIE + "\n" if poradie else "")
    return (f"Recall report\nTitle: {titulok}\nText: {_text(text, max_znakov)}\n\n{OTAZKA[stitok]}\n"
            f"{uvod}Options:\n" + "\n".join(riadky) + "\n\nAnswer with the letter of the correct option only.")


UVOD_PRIKLADY = ("The options are labels from an annotated dataset of recall reports. Each option shows how many "
                 "training reports have that label and the title of one of them. Choose the label that this "
                 "dataset would use for the report.")
UVOD_PORADIE = "The options are sorted by a trained classifier, most likely first; option A is its best guess."
# v1 = pôvodný prompt; v2 = + počet a príklad z train pri každej možnosti; v3 = v2 + informácia o poradí
PROMPTY = {"v1": {}, "v2": {"priklady": True}, "v3": {"priklady": True, "poradie": True}}
NEINFORMATIVNY = re.compile(r"FSIS|Recall Notification", re.IGNORECASE)


def priklady_tried(train):
    """Pre hazard a product: trieda -> (počet správ v train, príklad titulku). Príklad je titulok strednej
    dĺžky spomedzi správ s danou triedou; prednosť majú titulky, ktoré obsahujú celý názov triedy, potom
    jeho prvé slovo, potom ostatné bez „Recall Notification: FSIS-…“. Skrátený na 100 znakov; výber je
    deterministický a používa len train."""
    vystup = {}
    for lab in STITKY:
        d = {}
        for trieda, g in train.groupby(lab):
            tit = g["title"].astype(str).str.strip()
            slova = re.findall(r"[a-z]{3,}", trieda.lower())
            urovne = [tit[tit.str.lower().str.contains(re.escape(trieda.lower()))]]
            if slova:
                urovne.append(tit[tit.str.lower().str.contains(r"\b" + re.escape(slova[0]))])
            urovne += [tit[~tit.str.contains(NEINFORMATIVNY)], tit]
            vyber = next(u for u in urovne if len(u)).sort_values(key=lambda s: s.str.len(), kind="stable")
            d[trieda] = (len(g), re.sub(r"\s+", " ", vyber.iloc[len(vyber) // 2])[:100])
        vystup[lab] = d
    return vystup


def prompt_fn(verzia):
    """Funkcia (titulok, text, stitok, kandidati, max_znakov) -> prompt pre zvolenú verziu promptu."""
    nast = PROMPTY[verzia]
    pr = priklady_tried(nacitaj("train")) if nast.get("priklady") else None

    def f(titulok, text, stitok, kandidati, max_znakov):
        return prompt_pismena(titulok, text, stitok, kandidati, max_znakov,
                              priklady=pr[stitok] if pr else None, poradie=nast.get("poradie", False))
    return f


def odpoved_na_kandidata(odpoved, kandidati):
    """Backend openai: vráti vybraného kandidáta alebo None, ak odpoveď nezodpovedá žiadnej možnosti."""
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


def pravdepodobnosti_klasifikatora(skore):
    """Skóre kandidátov → rozdelenie pravdepodobností. Pri transformeri (E12kens: hazard, product)
    sú skóre pravdepodobnosti a len sa znormalizujú na k kandidátov; pri SVM (decision_function,
    môže byť záporné) sa použije softmax, ktorý nie je kalibrovaný."""
    s = np.asarray(skore, dtype=float)
    if np.isnan(s).any():
        raise ValueError("kandidáti nemajú skóre, spojenie s klasifikátorom nejde")
    if (s >= 0).all() and s.sum() > 0:
        return s / s.sum()
    e = np.exp(s - s.max())
    return e / e.sum()


def fuzia(p_klas, p_llm, w, eps=1e-4):
    """Index kandidáta s najväčším log p_klasifikátora + w · log p_LLM (w = 0 → kandidát č. 1)."""
    s = np.log(np.clip(p_klas, 1e-6, None)) + w * np.log(np.asarray(p_llm) + eps)
    return int(np.argmax(s))


class Cache:
    """Odpovede LLM v súbore JSONL, jeden záznam na riadok, zapisuje sa hneď po každej odpovedi."""

    def __init__(self, model):
        priecinok = os.path.join(VYSLEDKY, "llm_cache")
        os.makedirs(priecinok, exist_ok=True)
        self.cesta = os.path.join(priecinok, re.sub(r"[^\w.-]", "_", model) + ".jsonl")
        self.data = {}
        if os.path.exists(self.cesta):
            with open(self.cesta, encoding="utf-8") as f:
                for riadok in f:
                    if riadok.strip():
                        z = json.loads(riadok)
                        self.data[z["kluc"]] = z

    @staticmethod
    def kluc(model, text):
        return hashlib.sha256((model + "\n" + text).encode("utf-8")).hexdigest()

    def uloz(self, kluc, zaznam):
        zaznam = {"kluc": kluc, **zaznam, "cas": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.data[kluc] = zaznam
        with open(self.cesta, "a", encoding="utf-8") as f:
            f.write(json.dumps(zaznam, ensure_ascii=False) + "\n")


class HFVyberac:
    """Lokálny model (transformers): pravdepodobnosti písmen A, B, C … ako prvého tokenu odpovede."""

    def __init__(self, cesta):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            sys.exit("GPU nie je dostupná (torch.cuda.is_available() = False).")
        self.torch = torch
        print(f"načítavam model z {cesta} …", flush=True)
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(cesta)
        # Quadro RTX 4000 (Turing) nemá bf16, preto float16
        self.model = AutoModelForCausalLM.from_pretrained(cesta, dtype=torch.float16).to("cuda").eval()
        self.id_pismen = []
        for p in PISMENA:
            ids = self.tok.encode(p, add_special_tokens=False)
            assert len(ids) == 1, f"písmeno {p} nie je jeden token: {ids}"
            self.id_pismen.append(ids[0])
        self.info = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                     "transformers": __import__("transformers").__version__,
                     "pamat_GPU_GB": round(torch.cuda.memory_allocated() / 1024**3, 2)}
        print(f"model načítaný za {time.time() - t0:.0f} s, {self.info}", flush=True)

    def pravdepodobnosti(self, text, k):
        spravy = [{"role": "system", "content": SYSTEMOVA_HF}, {"role": "user", "content": text}]
        vstup = self.tok.apply_chat_template(spravy, add_generation_prompt=True, return_tensors="pt",
                                             return_dict=True).to("cuda")
        with self.torch.inference_mode():
            logity = self.model(**vstup, logits_to_keep=1).logits[0, -1].float()
        vyber = logity[self.id_pismen[:k]]
        if not self.torch.isfinite(vyber).all():
            raise RuntimeError("model vrátil NaN/inf (pretečenie vo float16?)")
        p = self.torch.softmax(vyber, dim=-1).tolist()
        # pravdepodobnosť, že model ako prvý token napíše niečo iné než jedno z písmen (kontrola promptu)
        p_vsetky = self.torch.softmax(logity, dim=-1)
        mimo = 1.0 - float(p_vsetky[self.id_pismen[:k]].sum())
        return [round(x, 6) for x in p], round(mimo, 6), int(vstup["input_ids"].shape[1])


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


def nazov_systemu(model, kandidati, prah, hierarchia, vaha=None, stitky_llm=tuple(STITKY), verzia="v1"):
    kratky = re.sub(r"[^\w]", "", model.split("/")[-1])[:20]
    system = f"E4_{kratky}_{kandidati.removeprefix('E3_')}"
    if verzia != "v1":
        system += f"_{verzia}"
    if prah > 0:
        system += f"_p{round(prah * 100):03d}"
    if vaha is not None:
        system += f"_f{round(vaha * 100):03d}"
    if tuple(stitky_llm) != tuple(STITKY):
        system += "_" + "_".join(s[:4] for s in stitky_llm)
    if hierarchia:
        system += "_hier"
    return system


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kandidati", default="E12kens", help="systém, ktorého kandidátov a predikcie použijeme")
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--backend", choices=["skuska", "hf", "openai"], default="skuska")
    ap.add_argument("--model", default="skuska", help="názov modelu (pri hf názov z Hugging Face, napr. "
                                                       "Qwen/Qwen2.5-3B-Instruct); je aj v kľúči cache")
    ap.add_argument("--model-cesta", help="backend hf: priečinok so stiahnutým modelom")
    ap.add_argument("--k", type=int, default=10, help="koľko kandidátov dostane LLM")
    ap.add_argument("--max-znakov", type=int, default=1500, help="koľko znakov textu správy ide do promptu")
    ap.add_argument("--prah", type=float, default=0.0,
                    help="backend hf: výber LLM sa použije len pri pravdepodobnosti aspoň prah (0 = vždy)")
    ap.add_argument("--fuzia", type=float, default=None, metavar="W",
                    help="backend hf: argmax log p_klasifikátora + W · log p_LLM namiesto samotného LLM")
    ap.add_argument("--stitky-llm", nargs="+", choices=STITKY, default=STITKY,
                    help="pri ktorých štítkoch sa výber LLM použije (ostatné ostanú zo základu)")
    ap.add_argument("--prompt", choices=list(PROMPTY), default="v1",
                    help="backend hf: verzia promptu (v2 a v3 potrebujú data/incidents_train.csv)")
    ap.add_argument("--ukaz-prompt", action="store_true", help="vypíše prompt pre prvú správu (hazard aj product)")
    ap.add_argument("--hierarchia", action="store_true",
                    help="ablácia: hazard-category odvodiť z vybraného hazardu (predvolene ostáva zo základu)")
    ap.add_argument("--limit", type=int, default=0, help="len prvých N správ (skúška), výsledok sa neuloží")
    ap.add_argument("--povolit-test", action="store_true", help="povolí beh na teste (až na konci)")
    a = ap.parse_args()
    if a.split == "test" and not a.povolit_test:
        sys.exit("Na test spúšťame až zmrazený systém na konci. Ak je to ten beh, pridaj --povolit-test.")
    if a.k > len(PISMENA):
        sys.exit(f"--k najviac {len(PISMENA)}")
    if (a.prah or a.fuzia is not None) and a.backend != "hf":
        sys.exit("--prah a --fuzia majú zmysel len pri backende hf (iba ten dáva pravdepodobnosti)")
    if a.prah and a.fuzia is not None:
        sys.exit("--prah a --fuzia naraz nepoužívaj")
    if a.prompt != "v1" and a.backend == "openai":
        sys.exit("--prompt v2/v3 je len pre backendy hf a skuska")
    vytvor_prompt = prompt_fn(a.prompt)

    data = nacitaj(a.split)
    zaklad = nacitaj_predikcie(a.kandidati, a.split).sort_values("id").reset_index(drop=True)
    kand = nacitaj_kandidatov(a.kandidati, a.split).sort_values(["stitok", "id", "poradie"])
    zoznamy = {lab: {i: g["trieda"].tolist()[:a.k] for i, g in kand[kand["stitok"] == lab].groupby("id")}
               for lab in STITKY}
    skore_kand = {lab: {i: g["skore"].to_numpy()[:a.k] for i, g in kand[kand["stitok"] == lab].groupby("id")}
                  for lab in STITKY}
    cache = Cache(a.model)
    n = a.limit or len(data)
    vyberac = None  # model sa načíta až pri prvej odpovedi, ktorá nie je v cache

    pred = zaklad.copy()
    statistika = {lab: {"volania": 0, "z_cache": 0, "nezhoda": 0, "zmenene": 0} for lab in STITKY}
    detail = []
    t0 = time.time()
    for i in range(n):
        for lab in STITKY:
            kandidati = zoznamy[lab][i]
            p = None
            if a.backend == "openai":
                text = prompt(data["title"][i], data["text"][i], lab, kandidati, a.max_znakov)
            else:
                text = vytvor_prompt(data["title"][i], data["text"][i], lab, kandidati, a.max_znakov)
            if a.ukaz_prompt and i == 0:
                print(f"\n----- prompt ({a.prompt}, {lab}, správa 0) -----\n{text}\n-----\n", flush=True)
            kluc = Cache.kluc(a.model, text)

            if kluc in cache.data:
                zaznam = cache.data[kluc]
                statistika[lab]["z_cache"] += 1
            else:
                statistika[lab]["volania"] += 1
                if a.backend == "skuska":
                    zaznam = {"odpoved": kandidati[0]}
                elif a.backend == "hf":
                    if vyberac is None:
                        if not a.model_cesta:
                            sys.exit("pri backende hf treba --model-cesta (priečinok so stiahnutým modelom)")
                        vyberac = HFVyberac(a.model_cesta)
                    pp, mimo, tokeny = vyberac.pravdepodobnosti(text, len(kandidati))
                    zaznam = {"odpoved": PISMENA[max(range(len(pp)), key=pp.__getitem__)],
                              "p": pp, "p_mimo_pismen": mimo, "tokeny": tokeny}
                else:
                    zaznam = {"odpoved": zavolaj_openai(a.model, text)}
                if a.backend != "skuska":
                    cache.uloz(kluc, zaznam)

            if a.backend == "hf":
                p = zaznam["p"]
                j = max(range(len(p)), key=p.__getitem__)  # samotný výber LLM (do detailu)
                if lab not in a.stitky_llm:
                    vyber = kandidati[0]
                elif a.fuzia is not None:
                    vyber = kandidati[fuzia(pravdepodobnosti_klasifikatora(skore_kand[lab][i]), p, a.fuzia)]
                else:
                    vyber = kandidati[j] if p[j] >= a.prah else kandidati[0]
            else:
                vyber = odpoved_na_kandidata(zaznam["odpoved"], kandidati)
                if vyber is None:
                    statistika[lab]["nezhoda"] += 1
                    vyber = kandidati[0]
                j = kandidati.index(vyber)
                if lab not in a.stitky_llm:
                    vyber = kandidati[0]

            if vyber != kandidati[0]:
                statistika[lab]["zmenene"] += 1
            pred.loc[i, lab] = vyber
            spravna = data[lab][i]
            detail.append({
                "id": i, "stitok": lab, "spravna": spravna,
                "poradie_spravnej": kandidati.index(spravna) + 1 if spravna in kandidati else 0,
                "kandidat_1": kandidati[0], "llm_poradie": j + 1, "llm_vyber": kandidati[j],
                "p_llm": p[j] if p else None, "p_kandidat_1": p[0] if p else None,
                "p_mimo_pismen": zaznam.get("p_mimo_pismen"), "tokeny": zaznam.get("tokeny"),
                "vysledok": vyber,
            })
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n} správ ({time.time() - t0:.0f} s)", flush=True)
    trvanie = time.time() - t0

    if a.hierarchia:
        mapa = mapa_hazard_kategoria(nacitaj("train"))
        pred["hazard-category"] = pred["hazard"].map(mapa).fillna(pred["hazard-category"])
    print("štatistika:", json.dumps(statistika, ensure_ascii=False))

    # Ako často LLM zmenil odpoveď základného systému a či to pomohlo (na úrovni jednotlivých správ)
    det = pd.DataFrame(detail)
    print("\nzmeny oproti kandidátovi č. 1 (počet správ):")
    for lab in STITKY:
        d = det[det["stitok"] == lab]
        zmena = d["vysledok"] != d["kandidat_1"]
        dobre_na_zle = (zmena & (d["kandidat_1"] == d["spravna"])).sum()
        zle_na_dobre = (zmena & (d["vysledok"] == d["spravna"])).sum()
        print(f"  {lab:8} zmenené {zmena.sum():4d} | zlé → správne {zle_na_dobre:4d} | správne → zlé "
              f"{dobre_na_zle:4d} | správna v top-{a.k}: {(d['poradie_spravnej'] > 0).mean():.1%} | "
              f"presnosť: základ {(d['kandidat_1'] == d['spravna']).mean():.1%}, "
              f"E4 {(d['vysledok'] == d['spravna']).mean():.1%}")

    vyrez = slice(0, n)
    s_zak = skore(data.iloc[vyrez], zaklad.iloc[vyrez])
    s_llm = skore(data.iloc[vyrez], pred.iloc[vyrez])
    if a.limit:
        print(f"\nskúška na prvých {n} správach (malá vzorka, len orientačne):")
    print(pd.DataFrame([{"systém": a.kandidati, **s_zak}, {"systém": "E4", **s_llm}]).round(4).to_string(index=False))
    if a.limit:
        return

    system = nazov_systemu(a.model, a.kandidati, a.prah, a.hierarchia, a.fuzia, a.stitky_llm, a.prompt)
    cesta = uloz_predikcie(pred, system, a.split)
    priecinok = os.path.join(VYSLEDKY, "llm")
    os.makedirs(priecinok, exist_ok=True)
    det.to_csv(os.path.join(priecinok, f"detail_{system}_{a.split}.csv"), index=False)
    log = {"system": system, "split": a.split, "backend": a.backend, "model": a.model, "k": a.k,
           "max_znakov": a.max_znakov, "prompt": a.prompt, "prah": a.prah, "fuzia": a.fuzia,
           "stitky_llm": a.stitky_llm,
           "hierarchia": a.hierarchia, "kandidati": a.kandidati,
           "statistika": statistika, "trvanie_s": round(trvanie, 1),
           "trvanie_s_na_volanie": round(trvanie / max(1, sum(s["volania"] for s in statistika.values())), 3),
           "python": platform.python_version(), "cas": time.strftime("%Y-%m-%d %H:%M:%S"),
           **({"prostredie": vyberac.info} if vyberac else {}),
           "skore": {k: round(v, 4) for k, v in s_llm.items()}}
    os.makedirs(os.path.join(VYSLEDKY, "logy"), exist_ok=True)
    with open(os.path.join(VYSLEDKY, "logy", f"{system}_{a.split}.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print("\nuložené:", os.path.relpath(cesta, KOREN),
          f"| ďalej: python src/vyhodnot.py --split {a.split} {system} --porovnaj {a.kandidati} --bootstrap 1000")


if __name__ == "__main__":
    main()
