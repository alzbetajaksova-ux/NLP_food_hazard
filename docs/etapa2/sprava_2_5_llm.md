# Koncept časti 2.5 do správy (výber z kandidátov pomocou LLM)

> Koncept textu napísal AI agent zo zápisníka experimentov a z výstupov behov (20. 11. 2026). Pred vložením do `dokumentacia_2etapa_kostra.docx` ho treba skontrolovať a prepísať vlastnými slovami. Čísla sú na **valid**; výsledok na teste sa doplní po jedinom záverečnom behu. Miesta `[overiť]` a `[doplniť]` treba dokončiť.

---

## 2.5 Výber z kandidátov pomocou LLM

Pri jemných štítkoch býva správna trieda medzi kandidátmi klasifikátora, ale nie na prvom mieste. Pri E12kens je správny produkt na valid na 1. mieste v 54,3 % správ a medzi 10 kandidátmi v 76,1 %; správny hazard na 1. mieste v 87,6 % a v top 10 v 97,0 %. Druhý krok preto dá jazykovému modelu (LLM) titulok, text správy a 10 kandidátov klasifikátora a nechá ho vybrať jedného. Kategórie (hazard-category, product-category) ostávajú z E12kens.

### Model

Použili sme **Qwen2.5-3B-Instruct** (Qwen Team, 2024) s otvorenými váhami, spustený lokálne na školskej GPU Quadro RTX 4000 (8 GB) na portáli datalab, knižnica transformers 5.23.0, float16. Dôvody:

- zmestí sa do 8 GB GPU bez kvantizácie (5,9 GB po načítaní) a celý valid spracuje za niekoľko minút;
- nevyžaduje platené API, dáta neopúšťajú školský server a výsledok nezávisí od zmien vzdialenej služby;
- pri lokálnom modeli máme prístup k pravdepodobnostiam odpovedí, nielen k vygenerovanému textu, čo využíva spojenie s klasifikátorom (nižšie).

Ako abláciu veľkosti modelu sme skúsili **Qwen2.5-7B-Instruct** v 4 bitoch (checkpoint `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`, kvantizácia NF4 z bitsandbytes podľa Dettmers et al., 2023, výpočty v bfloat16). Model 3B je pod licenciou Qwen Research License Agreement (Alibaba Cloud, 19. 9. 2024), ktorá povoľuje použitie len na nekomerčné, výskumné alebo hodnotiace účely; školský projekt do nich patrí. Váhy modelu v repozitári nešírime, ukladáme len jeho odpovede. Model 7B je pod licenciou Apache 2.0.

### Ako sa model pýta

Pre každú správu a každý štítok (hazard, product) dostane model jednu otázku s 10 možnosťami označenými písmenami A až J v poradí podľa klasifikátora. Písmená namiesto čísel sme zvolili preto, lebo každé je v tokenizéri jeden token (číslo 10 sa delí na dva). Model **negeneruje text**: urobíme jeden prechod modelom a z výstupu pre prvý token odpovede vezmeme pravdepodobnosti desiatich písmen, ktoré znormalizujeme (softmax cez tieto písmená). Samotný výber LLM je písmeno s najvyššou pravdepodobnosťou, čo zodpovedá generovaniu s teplotou 0. Postup je deterministický a **neplatná odpoveď nemôže nastať**, lebo sa vyberá vždy jedna z ponúknutých možností. Ako kontrolu, či model formát odpovede dodržiava, ukladáme aj pravdepodobnosť, že by ako prvý token napísal niečo iné než písmeno: v priemere je pod 0,0001 a najviac 0,053.

Text správy sa skráti na 1 500 znakov (po zlúčení medzier), prompt má v priemere 464 tokenov (190 až 851). Celý prompt (verzia v1), použitý s chat šablónou modelu:

Systémová správa:

```text
You classify food recall reports. You always answer with the letter of exactly one option from the given list.
```

Správa používateľa (príklad pre product, správa 282 z valid):

```text
Recall report
Title: Tiny Rebel Brewing Co. recalls bottles of Stay Puft Marshmallow Porter because lactose is not declared as being derived from milk
Text: Tiny Rebel Brewing Co. is recalling bottles of Stay Puft Marshmallow Porter because the label does not mention … [prvých 1 500 znakov textu]

Which food product is being recalled?
Options:
A. marshmalows
B. beer
C. alcoholic beverages
D. lime
E. butter
F. coffee drink
G. cashew
H. cereal
I. cereal bars
J. poppy

Answer with the letter of the correct option only.
```

Pri hazarde je otázka „Which food hazard is the reason for this recall?“. Na príklade vybral klasifikátor zložku spomenutú v texte (marshmalows), LLM vybral správny produkt (beer).

### Spojenie s klasifikátorom

Samotný výber LLM je výrazne horší než klasifikátor (tabuľka nižšie), lebo LLM často mení aj správne odpovede, pri ktorých si je klasifikátor istý. Preto obe rozdelenia spájame: pre každého kandidáta *c*

skóre(*c*) = log *p*_klasifikátor(*c*) + *w* · log(*p*_LLM(*c*) + 10⁻⁴)

a vyberie sa kandidát s najvyšším skóre (*p*_klasifikátor sú pravdepodobnosti E12kens znormalizované na 10 kandidátov, *w* = 0 je pôvodný E12kens). Váhu *w* a štítky, pri ktorých sa LLM použije, sme vybrali na valid opakovanou dvojnásobnou krížovou validáciou: valid sa 50-krát náhodne rozdelí na polovice, *w* sa vyberie na jednej a zisk sa zmeria na druhej. Priemer týchto ziskov je poctivejší odhad, než keď sa *w* vyberie aj vyhodnotí na celom valid. Najstabilnejší bol variant s LLM **len pri produkte** a *w* = 0,3: zisk ST2 +0,018 (2,5 až 97,5 % rozsah [+0,001; +0,034], kladný v 98 % delení). To je výsledný systém E4 (`E4_Qwen253BInstruct_E12kens_f030_prod`).

### Výsledky na valid a ablácie

| Variant (valid) | ST2 | ST2 − E12kens [95 % CI] |
| --- | --- | --- |
| E12kens (bez LLM) | 0,495 | – |
| LLM sám, 3B | 0,368 | −0,127 [−0,167; −0,097] |
| LLM sám, len ak *p*_LLM ≥ 0,9 | 0,407 | −0,088 [−0,128; −0,059] |
| LLM sám, prompt v2 (+ počty a príklady z train) | 0,292 | – |
| LLM sám, 7B v 4 bitoch | 0,425 | −0,070 [−0,116; −0,041] |
| **E4: spojenie, len product, *w* = 0,3 (3B)** | **0,517** | **+0,022 [+0,010; +0,036]** |
| spojenie, hazard aj product, *w* = 0,1 (3B) | 0,525 | +0,030 [+0,008; +0,045] |
| spojenie, len product, *w* = 0,2 (7B) | 0,522 | +0,027 [+0,015; +0,038] |

Intervaly sú z párového bootstrapu (1 000 výberov). Pri spojeniach je *w* vybrané na tom istom valid, preto sú čísla mierne optimistické; odhad z krížovej validácie je uvedený vyššie. ST1 sa nemení (0,802), lebo kategórie sa preberajú z E12kens.

Zistenia (podrobne v `results/etapa2/llm/analyza/`):

- **Kde LLM škodí.** Pri hazarde nahrádza konkrétne alergény, ktoré dataset používa („milk and products thereof“, 588 správ v train), všeobecnými triedami („undeclared constituent“ 7 správ, „allergens“ 13). V kategórii allergens je 103 zo 144 zmien zo správnej na zlú odpoveď. Model si je pritom príliš istý: keď dá odpovedi pravdepodobnosť nad 0,99, je správna len v 71 % (hazard) a 54 % (product). Preto nepomáha ani prah na istotu LLM.
- **Kde LLM pomáha.** Pri produkte, keď si klasifikátor nie je istý (pravdepodobnosť kandidáta 1 medzi 0,3 a 0,7): spojenie tu opraví 30 a pokazí 11 správ. Typicky klasifikátor vyberie zložku spomenutú v texte a LLM samotný produkt (peanuts → candies, ham → sandwiches). Pri škodách LLM volí synonymum alebo všeobecnejšiu triedu (chilled salads → salads, milk chocolates → chocolate), čo súvisí s nejednoznačnou granularitou produktov v datasete.
- **Prečo malý počet zmien stačí.** Opravy sú najmä vo vzácnych triedach s 1 až 2 správami vo valid, ktoré macro-F1 váži rovnako ako časté triedy. Čistých 20 opravených produktov zvýši product macro-F1 z 0,335 na 0,380.
- **Prompt.** Doplnenie počtu trénovacích správ a príkladu titulku ku každej možnosti (v2), resp. informácie o poradí kandidátov (v3) výber LLM zhoršilo (ST2 0,292 a 0,316) a v spojení neprinieslo viac než v1 (zisk z krížovej validácie +0,016 a +0,009 oproti +0,018). Model túto informáciu o konvenciách datasetu nevyužil.
- **Veľkosť modelu.** 7B model vyberá sám lepšie než 3B (0,425 oproti 0,368), ale v spojení je rozdiel oproti 3B len +0,005 [−0,007; +0,014] pri šesťnásobnom čase výpočtu. Ostali sme pri 3B.

### Náklady

Za API sme neplatili nič; náklad je čas školskej GPU. Model 3B má na disku 5,8 GB (stiahnutie asi 1,5 minúty), načítanie trvá 1,5 až 3 minúty a jedna otázka asi 0,28 s, celý valid (565 správ, 1 130 otázok) asi 5,5 minúty. Model 7B v 4 bitoch zaberá 5,2 GB, načítanie 55 s a otázka 1,9 s (celý valid 35 minút). Záverečný beh na teste (997 správ) s modelom 3B odhadujeme na 10 až 15 minút.

### Reprodukovateľnosť

- Kód: `src/e4_llm.py` (beh na GPU), `src/e4_vaha_cv.py` (výber váhy), `src/e4_analyza.py` (analýza zmien); notebooky z datalabu s výpismi v `notebooks/etapa2_e4*_datalab.ipynb`. Verzie balíčkov pre E4 sú pripnuté v `requirements_e4.txt` (torch 2.15.0, transformers 5.23.0, bitsandbytes 0.50.2, accelerate 1.16.0; Python 3.12.10).
- Model: `Qwen/Qwen2.5-3B-Instruct`, revízia `aa8e72537993ba99e69dfaafa59ed015b17504d1` (commit na Hugging Face); stiahnutý cez `huggingface_hub.snapshot_download`.
- **Všetky odpovede LLM sú uložené** v `results/etapa2/llm_cache/<model>.jsonl` (pravdepodobnosti všetkých 10 možností, počet tokenov, kľúč = SHA-256 z názvu modelu a celého promptu). Opakovaný beh `e4_llm.py` ich načíta bez GPU a dá presne rovnaké predikcie; zmena promptu alebo modelu vedie k novému kľúču, takže sa staré odpovede omylom nepoužijú. Výber váhy aj analýza bežia len z tejto cache na CPU.
- Pri každom behu sa do `results/etapa2/logy/` zapíše nastavenie, verzie, GPU, kvantizácia, pamäť a čas; po správach sa ukladá detail (kandidát 1, výber LLM, pravdepodobnosti) do `results/etapa2/llm/detail_*.csv`.
- Nový beh na GPU môže pri niektorých otázkach dať mierne iné pravdepodobnosti (nedeterminizmus výpočtov vo float16 na GPU), preto protokol reprodukovateľnosti pri E4 vyžaduje presnú zhodu výsledku z uložených odpovedí.

---

## Doplnky pre iné časti správy

**Kapitola 4, riadok E4 do tabuľky výsledkov na valid:**

| Systém (valid) | ST1 | ST2 | hazard-category | product-category | hazard | product |
| --- | --- | --- | --- | --- | --- | --- |
| E4: + LLM (Qwen2.5-3B, len product) | 0,802 | 0,517 | 0,853 | 0,755 | 0,655 | 0,361 |

**Kapitola 3, tabuľka experimentov:** riadok E4 môže ostať „E12kens + výber LLM z top-10 kandidátov / prínos druhého kroku“; ablácie E4: LLM sám, prah, prompt v2/v3, veľkosť modelu 3B/7B, spojenie len pri product alebo pri oboch.

**Kapitola 8, obmedzenia E4:** váha spojenia a štítky boli zvolené na valid (krížová validácia zmierňuje, ale neodstraňuje optimizmus); výsledok závisí od konkrétneho LLM; LLM pomáha len vtedy, keď je správna trieda medzi 10 kandidátmi (pri produkte 76 % správ).

**Zoznam literatúry, doplniť:**

- Qwen Team (2024). Qwen2.5 Technical Report. arXiv:2412.15115. [z pamäti, overiť]
- Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023. arXiv:2305.14314. [z pamäti, overiť] (len ak sa v správe spomenie 7B v 4 bitoch)
- Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. EMNLP 2020: System Demonstrations. [z pamäti, overiť]
- Karta modelu: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
