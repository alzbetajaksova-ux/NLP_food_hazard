# Zápisník experimentov (2. etapa)

Všetky čísla sú na **valid** (565 správ), ak nie je uvedené inak. Test sa spúšťa jediný raz na konci. Pri každom zázname je, čo sa vyberalo na valid, lebo taký výsledok je mierne optimistický.

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
