# Protokol nezávislej reprodukovateľnosti (šablóna)

Zadanie (bonus): osoba mimo tímu spustí projekt v čistom prostredí iba podľa odovzdanej dokumentácie a reprodukuje hlavné výsledky v stanovenej tolerancii. Tolerancie sú stanovené **vopred**, pred spustením. Organizuje osoba 2.

## Tolerancie (vopred)

| Výsledok | Ako sa reprodukuje | Tolerancia |
| --- | --- | --- |
| E0_svm, E0_svmvyber, E2kens, E12kens | `python src/zostav_systemy.py` (z uložených behov transformera) | presná zhoda skóre (rozdiel < 0,0005) |
| E1, E2, E1b (transformer) | nový tréning so semienkom 42 na GPU (notebook) | v rámci priemer ± 2 smerodajné odchýlky z našich 3 semienok |
| E4 (LLM) | z uložených odpovedí v `results/etapa2/llm_cache/` | presná zhoda skóre |
| Tabuľka výsledkov na teste | `src/vyhodnot.py --split test --povolit-test` nad uloženými predikciami | presná zhoda |

## Záznam

- **Kto:** [meno, vzťah k tímu: mimo tímu]
- **Kedy:** [dátum]
- **Prostredie:** [OS, verzia Pythonu, CPU/GPU, Colab áno/nie]
- **Z čoho postupoval:** README v commite [hash], nič iné.
- **Kroky a čas:** [čo spustil, koľko to trvalo]
- **Problémy a ako ich vyriešil:** [čo v dokumentácii chýbalo alebo bolo nejasné]

| Výsledok | Náš | Reprodukovaný | V tolerancii |
| --- | --- | --- | --- |
| [systém, metrika] | | | |

**Úpravy dokumentácie po teste:** [čo sme na základe zistení opravili v README, commit].
