# Zápisník experimentov (2. etapa)

Všetky čísla sú na **valid** (565 správ), ak nie je uvedené inak. Test sa spúšťa jediný raz na konci. Pri každom zázname je, čo sa vyberalo na valid, lebo taký výsledok je mierne optimistický. Záznamy sú chronologické; staršie závery sú prekonané novšími.

**Aktuálny stav (17. 11. 2026):** najlepší systém je `E12kens` (ST1 0,802, ST2 0,495). Kandidáti pre LLM sú `E12kens`. Pomocné pravidlá (E3) sa nepoužívajú. Medzikroky (`E0_svmtit`, ensemble, `E2k_s*`, E3) nie sú v gite; vytvorí ich `python src/zostav_systemy.py --ablacie` (E1ens: `python src/ensemble.py --systemy E1_robertabase_s42 E1_robertabase_s43 E1_robertabase_s44 --nazov E1ens`; `E2k_s*`: `python src/kombinacia.py --zaklad E0_svmvyber --z E2_robertabase_s42 --stitky hazard-category --nazov E2k_s42`).

| Dátum | Experiment | Výsledok na valid | Rozhodnutie |
| --- | --- | --- | --- |
| 15. 11. 2026 | E0: baseline SVM (title + text, balanced) z 1. etapy | ST1 0,736, ST2 0,416; správny produkt v top-1 / top-10: 40,9 % / 67,6 % | referencia; na teste overená zhoda s 1. etapou (0,7225 / 0,4097) |
| 15. 11. 2026 | E3 nad E0: hierarchia hazard → kategória | ST1 0,747 (+0,011, interval [−0,026; +0,047]) | nepreukázané, ale konzistentné predikcie |
| 15. 11. 2026 | E3 nad E0: produkt z titulku, ak je v top-k modelu (k z {3, 5, 10} vybrané na valid = 5) | ST2 0,426 (+0,011 [+0,000; +0,020]) | malý prínos na hranici |
| 15. 11. 2026 | kandidáti E3 (produkt z titulku na 1. mieste) | správny produkt v top-10: 71,0 % | lepší vstup pre LLM než E0 |
| 15. 11. 2026 | kNN: kandidáti z produktov najpodobnejších správ v train (TF-IDF slov / znakov, titulok / title + text) | samotné kNN top-10: 52 až 62 %; zmiešané s E3: 72,0 až 72,6 % | prínos 1 až 1,6 bodu, nepoužité (pridáva zložitosť) |
| 15. 11. 2026 | E0_svmtit: SVM s pridanými znakovými n-gramami (3 až 5) titulku | hazard-category 0,808, product-category 0,755, product 0,269 (lepšie); hazard 0,561 (horšie než 0,598) | znaky titulku pomáhajú pri kategóriách a produkte, škodia hazardu |
| 15. 11. 2026 | E0_svmvyber: pre každý štítok model lepší na valid (hazard z E0_svm, ostatné z E0_svmtit) | ST1 0,777 (+0,041 [+0,014; +0,071]), ST2 0,447 (+0,031 [+0,020; +0,040]) | zatiaľ najlepší systém bez GPU a LLM |
| 15. 11. 2026 | E3 nad E0_svmvyber | ST1 0,776, ST2 0,443 (hierarchia ani produkt z titulku už nepomáhajú) | predikcie z E0_svmvyber; kandidáti E3_svmvyber (správny produkt v top-10: 72,6 %) sú zatiaľ najlepší vstup pre LLM |

Zistenie pre správu: hierarchia a produkt z titulku pomáhajú slabšiemu modelu, ale ich prínos zmizne, keď model sám lepšie využíva titulok. Pri transformeri to treba overiť znova.

## Transformer (Colab, Tesla T4), 15. 11. 2026

RoBERTa-base, title + text, max. 256 tokenov, 4 epochy, lr 2e-5, batch 16, výber epochy na valid; jeden beh (4 štítky) asi 22 minút, 6 behov spolu asi 2,2 hodiny.

| Variant (valid, priemer ± sm. odchýlka cez semienka 42, 43, 44) | ST1 | ST2 | hazard-category | product-category | hazard | product |
| --- | --- | --- | --- | --- | --- | --- |
| E0_svm | 0,736 | 0,416 | 0,787 | 0,714 | 0,598 | 0,215 |
| E0_svmvyber | 0,777 | 0,447 | 0,808 | 0,755 | 0,598 | 0,269 |
| E1 (bez váh) | 0,729 ± 0,012 | 0,253 ± 0,002 | 0,784 ± 0,037 | 0,636 ± 0,017 | 0,422 ± 0,008 | 0,076 ± 0,008 |
| E2 (s váhami) | 0,769 ± 0,024 | 0,218 ± 0,036 | 0,824 ± 0,044 | 0,664 ± 0,010 | 0,437 ± 0,071 | 0,000 ± 0,000 |

Zistenia:
- Transformer je lepší len pri **hazard-category** (E2: 0,824 oproti 0,808 pri SVM), ale s veľkým rozptylom medzi semienkami (0,775 až 0,860).
- Pri jemných štítkoch zlyháva: product 0,08 (E1) a 0,00 (E2), hazard 0,42 až 0,44 oproti 0,60 pri SVM. Najlepšia epocha bola takmer vždy posledná, model sa teda nestihol doučiť; pri 1 022 triedach a váhach tried E2 skolaboval (správny produkt v top-10 len v 3 % správ).
- Kandidáti transformera pre produkt sú horší než SVM (top-10: 50,8 % pri E1 oproti 71,9 % pri E0_svmvyber), pre LLM preto zostávajú kandidáti SVM.

Kombinácia s pevným pravidlom určeným podľa priemeru (hazard-category z E2, ostatné z E0_svmvyber), systémy `E2k_s42/43/44`: ST1 0,794 / 0,763 / 0,802, priemer 0,786 ± 0,021, ST2 nezmenené 0,447. Oproti E0_svmvyber: +0,017 [−0,008; +0,047], −0,014 [−0,050; +0,048], +0,025 [+0,001; +0,057]. Prínos je malý a závisí od semienka, preto ho zatiaľ nepovažujeme za preukázaný.

Ďalší krok: E1b (notebook, bunka 9), dlhší tréning len pre hazard a product (10 epoch, vyšší lr hlavy, bez váh). Ak ani ten neprekoná SVM, záver je, že pri tomto množstve dát a počte tried je pri jemných štítkoch lepší lineárny model a transformer sa použije len pre hazard-category.

## Ensemble semienok (bez GPU), 15. 11. 2026

Pravdepodobnosti troch behov sa sčítajú (`src/ensemble.py`); pri hazard-category (10 tried) ide o presný priemer celých rozdelení.

| Systém (valid) | hazard-category | ST1 | ST2 |
| --- | --- | --- | --- |
| E2ens (3 semienka E2) | 0,853 (jednotlivé behy 0,775 až 0,860, priemer 0,824) | 0,796 | 0,247 |
| E1ens (3 semienka E1) | 0,778 | 0,744 | 0,256 |
| **E2kens**: hazard-category z E2ens, ostatné z E0_svmvyber | 0,853 | **0,802** | **0,447** |

E2kens oproti E0_svmvyber: ST1 +0,025 [+0,001; +0,056], lepší v 98 % výberov; oproti E0: ST1 +0,066 [+0,034; +0,104], ST2 +0,031 [+0,020; +0,040]. Ensemble odstraňuje závislosť od jedného semienka. Voľba E2 (nie E1 ani E1+E2) bola urobená na valid, preto je výsledok mierne optimistický. Zatiaľ najlepší systém: **E2kens**.

## E1b: dlhší tréning pre hazard a product (Colab T4), 16. 11. 2026

RoBERTa-base, len štítky hazard a product, 10 epoch, lr 3e-5, lr klasifikačnej hlavy 1e-3, bez váh tried; ostatné štítky doplnené z E0_svmvyber. Jeden beh asi 25 minút (2 štítky × asi 12,5 minúty).

| Systém (valid) | hazard | product | ST2 | najlepšia epocha (hazard / product) |
| --- | --- | --- | --- | --- |
| E0_svmvyber (SVM) | 0,598 | 0,269 | 0,447 | – |
| E1 (4 epochy, lr 2e-5) priemer | 0,422 | 0,076 | 0,253 | 3 až 4 / 4 |
| E1b s42 / s43 / s44 | 0,657 / 0,645 / 0,644 | 0,307 / 0,325 / 0,328 | 0,484 / 0,484 / 0,493 | 8 / 9, 9 / 8, 10 / 7 |
| E1bens (ensemble 3 semienok) | 0,655 | 0,334 | 0,495 | – |

Dlhší tréning a vyšší lr hlavy vyriešili problém z E1/E2: transformer je teraz lepší než SVM aj pri hazarde (+0,057) a produkte (+0,065). Rozptyl medzi semienkami je malý (ST2 0,484 až 0,493). Správny produkt v top-10 kandidátoch: 76,1 % (SVM 71,9 %), hazard na 1. mieste 87,6 % (SVM 82,7 %).

**E12kens** (hazard-category z E2ens, product-category z E0_svmvyber, hazard a product z E1bens): na valid ST1 0,802 a ST2 0,495. Oproti E0: ST1 +0,066 [+0,034; +0,104], ST2 +0,079 [+0,047; +0,112]; oproti E2kens: ST2 +0,048 [+0,020; +0,083], lepší v 100 % výberov. **Zatiaľ najlepší systém.** Pravidlá kombinácie boli určené na valid, preto je výsledok mierne optimistický; potvrdí ho test.

E3 nad E12kens: hierarchia zhorší ST1 (0,802 → 0,788, lebo hazard-category z transformera je lepšia než odvodená z hazardu) a produkt z titulku zhorší produkt (0,334 → 0,321). Kandidáti E3 majú správny produkt v top-10 v 77,0 % (E12kens 76,1 %). Záver: pri silnom modeli pomocné pravidlá nepomáhajú; E3 sa nepoužije.

## E4: výber LLM z kandidátov E12kens (datalab), 20. 11. 2026

Qwen2.5-3B-Instruct vo float16 na Quadro RTX 4000 (8 GB, datalab; torch 2.15.0+cu132, transformers 5.23.0), model v GPU 5,85 GB. Pre hazard a product dostane LLM titulok, prvých 1 500 znakov textu a 10 kandidátov E12kens označených A až J. Nič sa negeneruje: berie sa pravdepodobnosť prvého tokenu odpovede pre písmená A až J (odpoveď je vždy platná, beh je deterministický). hazard-category a product-category ostávajú z E12kens. Celý valid (1 130 otázok) asi 0,2 s na otázku, spolu 5 minút vrátane načítania modelu. Odpovede sú v `results/etapa2/llm_cache/`, detail po správach v `results/etapa2/llm/`.

| Systém (valid) | LLM zmenil hazard / product | hazard | product | ST2 | ST2 − E12kens [95 % CI] |
| --- | --- | --- | --- | --- | --- |
| E12kens | – | 0,655 | 0,334 | 0,495 | – |
| E4 (LLM vždy) | 191 / 302 | 0,459 | 0,269 | 0,368 | −0,127 [−0,167; −0,097] |
| E4, prah istoty 0,7 | 172 / 260 | 0,480 | 0,290 | 0,385 | −0,110 [−0,151; −0,082] |
| E4, prah istoty 0,9 | 156 / 225 | 0,498 | 0,307 | 0,407 | −0,088 [−0,128; −0,059] |

Samotný výber LLM zhoršuje ST2 vo všetkých variantoch (presnosť top-1: hazard 87,6 % → 64,1 %, product 54,3 % → 42,7 %). ST1 sa nemení (0,802).

Analýza chýb (detail po správach):
- **Hazard:** 144 zmien zo správnej odpovede na zlú, 11 opačne. Takmer všetky zlé zmeny vyberú všeobecnú triedu namiesto konkrétnej, ktorú používa dataset: konkrétny alergén (milk, gluten, soybeans, eggs … „and products thereof“) → „undeclared constituent“ (50) alebo „allergens“ (36); listeria monocytogenes → microbiological contamination. LLM nepozná konvenciu označovania datasetu.
- **Product:** 113 zlých zmien, 47 dobrých; chyby sú takmer synonymá a rozdiely v úrovni podrobnosti (soup / soups, biscuits / cookies, corn / pop corn, chicken based products / chicken preparations).
- **Kalibrácia:** pri istote LLM nad 0,99 je jeho výber správny len v 71 % (hazard) a 54 % (product), preto prah istoty nepomáha.

Predbežne (váha vybraná na tých istých dátach, overené nižšie krížovou validáciou): spojenie log p_klasifikátora + w · log p_LLM zlepší ST2 na 0,525 (w = 0,1, oba štítky) a 0,517 (w = 0,3, len product); pri produkte je zisk stabilný pre w od 0,1 do 0,5, pri hazarde len pri w = 0,1.

### E4 krok A: spojenie klasifikátora a LLM, 20. 11. 2026

Bez GPU, z uložených odpovedí. Pre každého kandidáta log p_klasifikátora + w · log p_LLM, vyberie sa najväčší; p_klasifikátora sú pravdepodobnosti E1bens znormalizované na 10 kandidátov (`e4_llm.py --fuzia w`, `--stitky-llm`). Váha w z {0; 0,05; 0,1; 0,2; 0,3; 0,5; 0,75; 1} sa vyberá opakovanou dvojnásobnou krížovou validáciou na valid (`src/e4_vaha_cv.py`, 50 delení na polovice, seed 42): w sa zvolí na jednej polovici, zisk ST2 oproti E12kens sa meria na druhej.

| LLM pri | ST2 zisk (CV, priemer) | rozsah 2,5–97,5 % | zisk > 0 | zvolené w |
| --- | --- | --- | --- | --- |
| hazard aj product | +0,018 | [−0,007; +0,041] | 95 % | 0,1 |
| **len product** | **+0,018** | **[+0,001; +0,034]** | **98 %** | **0,3** |
| len hazard | +0,005 | [−0,017; +0,024] | 69 % | 0,1 |

Systémy s w zvoleným na celom valid (párový bootstrap 1 000 oproti E12kens; čísla sú mierne optimistické, poctivý odhad je CV vyššie):

| Systém (valid) | hazard | product | ST2 | ST2 − E12kens [95 % CI] | zmeny produktu (zlé → správne / správne → zlé) |
| --- | --- | --- | --- | --- | --- |
| E12kens | 0,655 | 0,334 | 0,495 | – | – |
| `E4_Qwen253BInstruct_E12kens_f030_prod` | 0,655 | 0,361 | 0,517 | +0,022 [+0,010; +0,036] | 38 / 18 |
| `E4_Qwen253BInstruct_E12kens_f010` | 0,683 | 0,355 | 0,525 | +0,030 [+0,008; +0,045] | 19 / 7 (hazard 7 / 6) |

Zistenia:
- S malou váhou LLM pomáha: mení odpoveď len tam, kde si klasifikátor nie je istý. Pri produkte sa presnosť top-1 zvýši z 54,3 % na 57,9 %.
- Pri hazarde je zisk neistý: macro-F1 hazardu stúpne na 0,683, hoci správnych odpovedí pribudne len o 1 (7 zlých → správne, 6 opačne); zmena je v zriedkavých triedach a v CV sa nepotvrdila.
- **Rozhodnutie:** hlavný variant E4 je `E4_Qwen253BInstruct_E12kens_f030_prod` (LLM len pri produkte, w = 0,3): zisk je stabilný pre w od 0,1 do 0,5 a v CV kladný v 98 % delení. Variant `f010` (oba štítky) ostáva ako ablácia.

### E4 krok B: prompt s počtami a príkladmi z train, 20. 11. 2026

Hypotéza z analýzy chýb v1: LLM nepozná konvencie datasetu (v train má „undeclared constituent“ 7 správ, „allergens“ 13, „milk and products thereof“ 588). Prompt **v2** pridáva ku každej možnosti počet trénovacích správ s touto triedou a jeden ich titulok (strednej dĺžky, prednostne obsahujúci názov triedy; `priklady_tried` v `e4_llm.py`), **v3** navyše informáciu, že možnosti zoradil klasifikátor a A je jeho najlepší tip. Prompt je asi o 300 tokenov dlhší; beh celého valid asi 4 minúty bez načítania modelu.

| Prompt (valid) | presnosť LLM top-1 hazard / product | ST2 samotný LLM | ST2 zisk spojenia (CV), len product | ST2 zisk spojenia (CV), oba štítky |
| --- | --- | --- | --- | --- |
| v1 | 64,1 % / 42,7 % | 0,368 | +0,018 [+0,001; +0,034], 98 % | +0,018 [−0,007; +0,041], 95 % |
| v2 (+ počty a príklady) | 54,5 % / 31,0 % | 0,292 | +0,016 [+0,003; +0,027], 98 % | +0,020 [−0,013; +0,046], 92 % |
| v3 (+ poradie) | 61,9 % / 35,6 % | 0,316 | +0,009 [−0,008; +0,021], 93 % | +0,006 [−0,015; +0,027], 75 % |

(E12kens: presnosť top-1 87,6 % / 54,3 %, ST2 0,495.)

Zistenia:
- Počty a príklady samotný výber LLM **zhoršili**. Model ich nevyužíva: pri zmene odpovede volí triedu početnejšiu v train než kandidát A len v 13 % (hazard) a 35 % (product) prípadov, všeobecné triedy volí ďalej („undeclared constituent“ 59-krát pri v2 oproti 50 pri v1) a pribudli nové („foreign bodies“ 29). Dlhší prompt zrejme 3B model skôr mätie.
- Informácia o poradí (v3) samotný LLM o niečo zlepší oproti v2, no v spojení s klasifikátorom pomáha menej než v1: model sa viac drží A, takže pridáva menej novej informácie.
- Hypotéza o konvenciách datasetu sa potvrdila v analýze chýb, ale model veľkosti 3B ju z metainformácie v prompte nedokáže využiť.
- **Rozhodnutie:** hlavný variant E4 ostáva `E4_Qwen253BInstruct_E12kens_f030_prod` (prompt v1, LLM len pri produkte, w = 0,3). v2 a v3 sú ablácie promptu. Ďalšie varianty promptu na valid neskúšame, aby výber nebol príliš optimistický.

### E4 krok C: väčší model Qwen2.5-7B-Instruct v 4 bitoch, 20. 11. 2026

Ablácia veľkosti modelu: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (bitsandbytes NF4, double quant), notebook `notebooks/etapa2_e4c_7b_datalab.ipynb`. Checkpoint má vlastný `quantization_config` s výpočtami v **bfloat16**, ktorý má prednosť pred nastavením v skripte (varovanie transformers pri načítaní; overené výpisom `config.json` na datalabe). Turing bfloat16 nemá natívne, preto je beh pomalší. V logoch behov v1 a v2 bol pôvodne nesprávny text „výpočty float16“ a ručne sa opravil podľa `config.json`; od commitu 918b5b7 skript zapisuje skutočné nastavenie. Rovnaké kandidáty, prompty a nastavenie ako pri 3B, len valid.

| Model, prompt (valid) | čas na otázku | GPU pamäť | presnosť LLM top-1 hazard / product | ST2 samotný LLM | ST2 zisk spojenia (CV), len product | ST2 zisk spojenia (CV), oba štítky |
| --- | --- | --- | --- | --- | --- | --- |
| 3B fp16, v1 | 0,31 s | 5,9 GB (po načítaní) | 64,1 % / 42,7 % | 0,368 | +0,018 [+0,001; +0,034], 98 % | +0,018 [−0,007; +0,041], 95 % |
| 7B 4-bit, v1 | 1,92 s | 5,2 GB (max 5,5) | 61,4 % / 48,1 % | 0,425 | +0,023 [+0,004; +0,037], 99 % | +0,021 [−0,006; +0,043], 92 % |
| 7B 4-bit, v2 | 3,13 s | 5,2 GB (max 5,7) | 60,5 % / 43,0 % | 0,376 | +0,019 [+0,002; +0,031], 99 % | +0,024 [−0,006; +0,051], 96 % |

Spojenia s váhou vybranou na celom valid (optimistické), bootstrap 1000 oproti E12kens (ST2 0,4949):

| Systém | w | ST2 | rozdiel [95 % CI] |
| --- | --- | --- | --- |
| `E4_Qwen253BInstruct_E12kens_f030_prod` (hlavný) | 0,3 | 0,5171 | +0,022 [+0,010; +0,036] |
| `E4_Qwen253BInstruct_E12kens_f010` | 0,1 | 0,5247 | +0,030 [+0,008; +0,045] |
| `E4_Qwen257BInstructbnb4_E12kens_f020_prod` | 0,2 | 0,5224 | +0,027 [+0,015; +0,038] |
| `E4_Qwen257BInstructbnb4_E12kens_f010` | 0,1 | 0,5273 | +0,032 [+0,010; +0,047] |
| `E4_Qwen257BInstructbnb4_E12kens_v2_f010` | 0,1 | 0,5323 | +0,037 [+0,013; +0,054] |

Priamo 7B oproti hlavnému 3B (`..._f030_prod`): `7B f020_prod` +0,005 [−0,007; +0,014], `7B f010` +0,010 [−0,012; +0,026]. Ani jeden rozdiel nie je preukázateľný.

Zistenia:
- Samotný 7B je menej zlý ako 3B (ST2 0,425 oproti 0,368), ale stále o 0,070 [−0,116; −0,041] horší než E12kens. Zlepšenie je pri produkte (presnosť top-1 48,1 % oproti 42,7 %), pri hazarde je 7B o niečo horší.
- Chyba so všeobecnými triedami hazardu ostáva, len sa presunula: zo 164 zmien správne → zlé je 92 na „allergens“ a 32 na „undeclared constituent“ (3B: 36 a 50). Pri produkte je zmien správne → zlé menej (84 oproti 113).
- Kalibrácia sa nezlepšila: pri p > 0,99 je 7B správny v 70 % (hazard) a 56 % (product), 3B v 71 % a 54 %.
- Prompt v2 ani 7B model nevyužíva: samotný výber je horší než pri v1. Výsledok spojenia v2 pri oboch štítkoch je najvyšší, ale rozdiel oproti v1 je v rámci šumu CV.
- 7B je asi 6-krát pomalší (1,9 s na otázku pri v1, celý valid 35 minút). Zisk v spojení je o 0,005 vyšší a nie je preukázateľný.
- **Rozhodnutie:** hlavný variant E4 ostáva `E4_Qwen253BInstruct_E12kens_f030_prod`. 7B je ablácia veľkosti modelu: väčší model sám vyberá lepšie, ale v spojení s klasifikátorom prínos nezvýši preukázateľne.
