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
