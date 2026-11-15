"""E1 / E2: doladenie transformera (napr. RoBERTa) pre všetky štyri štítky.

Pre každý štítok sa doladí samostatný model na title + text. Po každej epoche sa spočíta
macro-F1 na valid a ponechá sa najlepšia epocha (výber len na valid). Uloží sa:
  results/etapa2/predikcie/<system>_{valid,test}.csv   predikcie
  results/etapa2/kandidati/<system>_{valid,test}.csv   top-10 tried s pravdepodobnosťou
  results/etapa2/logy/<system>.json                    nastavenia, najlepšia epocha, F1 na valid, čas

E1 = bez váh tried, E2 = s váhami tried (--vahy). Každý variant spúšťame s 3 semienkami (42, 43, 44).
Príklad (GPU):  python src/e12_transformer.py --model roberta-base --seed 42
                python src/e12_transformer.py --model roberta-base --seed 42 --vahy
Rýchla skúška na CPU: --vzorka 200 --epochy 1 (výsledky sa vtedy neukladajú do results/).
"""
import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spolocne import (KOREN, LABELY, TOP_K, VYSLEDKY, nacitaj, uloz_kandidatov,  # noqa: E402
                      uloz_predikcie, vstup)


def nastav_semienko(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def zakoduj(tokenizer, texty, max_len):
    return tokenizer(list(texty), truncation=True, max_length=max_len)


def davky(kodovanie, y, batch, shuffle, seed, tokenizer):
    polozky = [{"input_ids": kodovanie["input_ids"][i], "attention_mask": kodovanie["attention_mask"][i],
                **({"labels": int(y[i])} if y is not None else {})} for i in range(len(kodovanie["input_ids"]))]
    g = torch.Generator()
    g.manual_seed(seed)

    def spoj(b):
        return tokenizer.pad(b, return_tensors="pt")
    return DataLoader(polozky, batch_size=batch, shuffle=shuffle, generator=g, collate_fn=spoj)


@torch.no_grad()
def pravdepodobnosti(model, loader, zariadenie):
    model.eval()
    vystup = []
    for b in loader:
        b = {k: v.to(zariadenie) for k, v in b.items() if k != "labels"}
        with torch.autocast(device_type=zariadenie.type, enabled=zariadenie.type == "cuda"):
            vystup.append(torch.softmax(model(**b).logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(vystup)


def dolad_stitok(lab, a, data, zariadenie, tokenizer):
    triedy = np.array(sorted(data["train"][lab].unique()))
    index = {t: i for i, t in enumerate(triedy)}
    y_tr = data["train"][lab].map(index).values
    kod = {s: zakoduj(tokenizer, vstup(data[s]), a.max_dlzka) for s in data}

    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=len(triedy), ignore_mismatched_sizes=True).to(zariadenie)
    vahy = None
    if a.vahy:
        w = compute_class_weight("balanced", classes=np.arange(len(triedy)), y=y_tr)
        vahy = torch.tensor(w, dtype=torch.float, device=zariadenie)
    strata = torch.nn.CrossEntropyLoss(weight=vahy)

    tren = davky(kod["train"], y_tr, a.batch, True, a.seed, tokenizer)
    ostatne = {s: davky(kod[s], None, a.batch * 2, False, a.seed, tokenizer) for s in ["valid", "test"]}
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    krokov = len(tren) * a.epochy
    plan = get_linear_schedule_with_warmup(opt, int(0.06 * krokov), krokov)
    scaler = torch.amp.GradScaler(enabled=zariadenie.type == "cuda")

    najlepsie = {"f1": -1.0, "epocha": 0, "stav": None}
    for ep in range(1, a.epochy + 1):
        model.train()
        for b in tren:
            b = {k: v.to(zariadenie) for k, v in b.items()}
            y = b.pop("labels")
            with torch.autocast(device_type=zariadenie.type, enabled=zariadenie.type == "cuda"):
                loss = strata(model(**b).logits.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            plan.step()
        p_val = pravdepodobnosti(model, ostatne["valid"], zariadenie)
        f1 = f1_score(data["valid"][lab], triedy[p_val.argmax(1)], average="macro", zero_division=0)
        print(f"  {lab}: epocha {ep}, macro-F1 valid {f1:.4f}")
        if f1 > najlepsie["f1"]:
            najlepsie = {"f1": f1, "epocha": ep, "stav": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})}

    model.load_state_dict(najlepsie["stav"])
    pravd = {s: pravdepodobnosti(model, ostatne[s], zariadenie) for s in ["valid", "test"]}
    del model
    if zariadenie.type == "cuda":
        torch.cuda.empty_cache()
    return triedy, pravd, {"najlepsia_epocha": najlepsie["epocha"], "f1_valid": round(najlepsie["f1"], 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="roberta-base", help="názov modelu z Hugging Face alebo cesta")
    ap.add_argument("--vahy", action="store_true", help="váhy tried 'balanced' v strate (E2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochy", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-dlzka", type=int, default=256, help="max. tokenov (titulok je na začiatku)")
    ap.add_argument("--vzorka", type=int, default=0, help="len prvých N správ z každého splitu (skúška)")
    ap.add_argument("--system", help="názov systému (predvolene E1/E2_<model>_s<seed>)")
    a = ap.parse_args()

    nastav_semienko(a.seed)
    zariadenie = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kratky = os.path.basename(a.model.rstrip("/")).replace("-", "").replace("_", "")
    system = a.system or f"{'E2' if a.vahy else 'E1'}_{kratky}_s{a.seed}"
    print(f"systém {system}, zariadenie {zariadenie}")

    data = {s: nacitaj(s) for s in ["train", "valid", "test"]}
    if a.vzorka:
        data = {s: d.head(a.vzorka).reset_index(drop=True) for s, d in data.items()}
    tokenizer = AutoTokenizer.from_pretrained(a.model)

    pred = {s: pd.DataFrame({"id": np.arange(len(data[s]))}) for s in ["valid", "test"]}
    kand = {s: [] for s in ["valid", "test"]}
    log = {"system": system, "nastavenia": vars(a), "zariadenie": str(zariadenie), "stitky": {}}
    t0 = time.time()
    for lab in LABELY:
        t = time.time()
        triedy, pravd, info = dolad_stitok(lab, a, data, zariadenie, tokenizer)
        log["stitky"][lab] = {**info, "sekundy": round(time.time() - t)}
        for s in ["valid", "test"]:
            poradie = np.argsort(-pravd[s], axis=1)[:, :TOP_K]
            n = len(data[s])
            pred[s][lab] = triedy[poradie[:, 0]]
            kand[s].append(pd.DataFrame({
                "id": np.repeat(np.arange(n), poradie.shape[1]), "stitok": lab,
                "poradie": np.tile(np.arange(1, poradie.shape[1] + 1), n),
                "trieda": triedy[poradie].ravel(),
                "skore": np.round(np.take_along_axis(pravd[s], poradie, axis=1).ravel(), 5)}))
    log["sekundy_spolu"] = round(time.time() - t0)

    if a.vzorka:
        print("skúška na vzorke hotová, výsledky sa neukladajú:", json.dumps(log["stitky"], ensure_ascii=False))
        return
    for s in ["valid", "test"]:
        uloz_predikcie(pred[s], system, s)
        uloz_kandidatov(pd.concat(kand[s]), system, s)
    os.makedirs(os.path.join(VYSLEDKY, "logy"), exist_ok=True)
    with open(os.path.join(VYSLEDKY, "logy", f"{system}.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print("uložené:", system, "| ďalej: python src/vyhodnot.py --split valid", system, "--porovnaj E0_svm")


if __name__ == "__main__":
    main()
