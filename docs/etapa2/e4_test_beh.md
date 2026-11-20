# E4: jediný záverečný beh na teste (odovzdanie pre osobu 1)

Systém E4 je **zmrazený**. Všetko nižšie bolo vybrané len na valid (zápisník, časti „E4“ z 20. 11. 2026). Na teste sa spustí raz, spolu s ostatnými zmrazenými systémami, a potom sa už nič nemení.

## Kto čo robí

| Krok | Kto | Kde | Potrebuje GPU |
| --- | --- | --- | --- |
| 0. potvrdiť, že všetky systémy sú zmrazené (nič sa už neladí) | osoba 1 | – | nie |
| 1. beh modelu na teste (notebook nižšie) | **osoba 2** | datalab | áno |
| 2. rozbaliť `vysledky_e4_test.zip`, commitnúť predikcie a odpovede, odovzdať cez git (vetva `etapa2-llm`) | osoba 2 | vlastný PC | nie |
| 3. vyhodnotenie všetkých systémov na teste, tabuľka do správy | **osoba 1** | ľubovoľný počítač | nie |
| 4. (voliteľne) kontrola, že predikcie E4 sa z uložených odpovedí vytvoria presne rovnako | osoba 1 | ľubovoľný počítač | nie |

Beh na GPU robí osoba 2, lebo na datalabe má stiahnutý presne ten model (revízia nižšie) a uložené odpovede z ladenia, takže test beží na tej istej GPU a s tými istými verziami ako ladenie na valid. Osoba 1 GPU ani model nepotrebuje: vyhodnotenie aj kontrola bežia z uložených predikcií a odpovedí za pár sekúnd. Beh na inej GPU (napr. Colab) by bol možný, ale pravdepodobnosti modelu by mohli vyjsť minimálne inak a prostredie by sa líšilo od ladenia.

## Zmrazené nastavenie

| Čo | Hodnota | Prečo (valid) |
| --- | --- | --- |
| systém | `E4_Qwen253BInstruct_E12kens_f030_prod` | najstabilnejší zisk v krížovej validácii |
| kandidáti a základ | E12kens, top 10 | najlepší systém bez LLM |
| model | `Qwen/Qwen2.5-3B-Instruct`, revízia `aa8e72537993ba99e69dfaafa59ed015b17504d1`, float16 | 7B nebol preukázateľne lepší a je 6-krát pomalší |
| prompt | v1, text skrátený na 1 500 znakov | v2 a v3 nepomohli |
| spojenie | log p_klasifikátor + 0,3 · log(p_LLM + 10⁻⁴), LLM **len pri product** | CV: +0,018 [+0,001; +0,034], kladné v 98 % delení |
| hazard, hazard-category, product-category | bez zmeny z E12kens | LLM pri hazarde škodí; hierarchia zhoršila ST1 |
| výsledok na valid | ST1 0,802, ST2 0,517 (E12kens 0,495; +0,022 [+0,010; +0,036]) | |

Ako ablácie sa na teste spočítajú len z uložených odpovedí (bez ďalšieho behu modelu): `E4_Qwen253BInstruct_E12kens` (LLM sám) a `E4_Qwen253BInstruct_E12kens_f010` (spojenie pri oboch štítkoch, w = 0,1). Ablácie promptu (v2, v3) a modelu 7B ostávajú len na valid.

## 1. Beh na GPU (datalab), asi 20 minút

Notebook `notebooks/etapa2_e4_test_datalab.ipynb`. Spúšťa ho osoba 2 **až keď osoba 1 potvrdí, že sú zmrazené všetky systémy**. V bunke 6 sa vypíše skóre na teste; po ňom sa nastavenie E4 už nemení. Notebook pred testom overí, že:

- model má rovnakú revíziu, aká sa použila pri ladení;
- hlavný systém na valid z uložených odpovedí dá presne ST2 0,5171, takže kód, dáta, kandidáti aj odpovede sú tie isté;
- kandidát č. 1 na teste je predikcia E12kens a každá správa má 10 kandidátov.

Beh zapíše odpovede modelu na test do `results/etapa2/llm_cache/Qwen_Qwen2.5-3B-Instruct.jsonl` a výsledky zbalí do `vysledky_e4_test.zip`. Obsah zipu sa rozbalí do koreňa repozitára:

- `results/etapa2/predikcie/E4_Qwen253BInstruct_E12kens{_f030_prod,,_f010}_test.csv`
- `results/etapa2/llm/detail_E4_…_test.csv`, `results/etapa2/logy/E4_…_test.json`
- doplnená cache odpovedí

Na datalab treba nahrať (cesty od koreňa repozitára): `src/e4_llm.py`, `notebooks/etapa2_e4_test_datalab.ipynb`, `data/incidents_test.csv`, `results/etapa2/kandidati/E12kens_test.csv`, `results/etapa2/predikcie/E12kens_test.csv`. Ostatné súbory tam už sú z predchádzajúcich behov.

## 2. Vyhodnotenie (osoba 1, bez GPU, raz pre všetky systémy)

```bash
python src/vyhodnot.py --split test --povolit-test \
  E0_svm E0_svmvyber E2kens E12kens E4_Qwen253BInstruct_E12kens_f030_prod \
  --porovnaj E0_svm --bootstrap 1000 --uloz results/etapa2/vyhodnotenie_test.csv
# prínos druhého kroku a ablácie E4 oproti E12kens
python src/vyhodnot.py --split test --povolit-test \
  E4_Qwen253BInstruct_E12kens_f030_prod E4_Qwen253BInstruct_E12kens E4_Qwen253BInstruct_E12kens_f010 \
  --porovnaj E12kens --bootstrap 1000 --uloz results/etapa2/llm/vyhodnotenie_test.csv
```

Analýzu, kde LLM pomáha a kde škodí, na teste nerobíme: skript `src/e4_analyza.py` je len pre valid, aby sa z testu nič neladilo.

## 3. Reprodukcia bez GPU (osoba 1, voliteľne)

Po stiahnutí vetvy s výsledkami behu (stačia balíčky z `requirements.txt`; torch, transformers ani model netreba). Keď sú odpovede v cache, ten istý príkaz dá presne rovnaké predikcie bez GPU (model sa načíta, len ak odpoveď v cache chýba):

```bash
python src/e4_llm.py --backend hf --model Qwen/Qwen2.5-3B-Instruct --split test \
  --kandidati E12kens --k 10 --max-znakov 1500 --prompt v1 --fuzia 0.3 --stitky-llm product --povolit-test
```

## 4. Do správy

- kapitola 4: riadok E4 do tabuľky na teste (z `results/etapa2/vyhodnotenie_test.csv`), rozdiel oproti E0 aj E12kens;
- časť 2.5 (`docs/etapa2/sprava_2_5_llm.md`): jedna veta s výsledkom na teste; ak sa výsledok líši od valid, napísať to tak, ako je, bez úprav nastavenia.
