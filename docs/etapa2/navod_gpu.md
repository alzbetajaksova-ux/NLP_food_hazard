# Návod: GPU pre transformer

GPU treba len na doladenie transformera (E1, E2, E1b). Vlastný výkonný počítač netreba: Google Colab aj Kaggle bežia v prehliadači. Ostatné časti projektu bežia na bežnom notebooku.

Overené k 15. 11. 2026 zo stránok uvedených na konci. Limity sa menia, preto pred začatím pozri aktuálny stav vo svojom účte.

## Možnosť 1: Google Colab (odporúčaná)

Čo treba vedieť:
- GPU je zadarmo, ale **nie je zaručené**; typ GPU sa mení (používatelia hlásia najčastejšie T4).
- Jedna relácia beží **najviac 12 hodín**, pri nečinnosti sa odpojí skôr. Google presné limity nezverejňuje.
- Po odpojení sa súbory vo virtuálnom stroji stratia, preto notebook priebežne ukladá výsledky na Google Drive.

Postup:
1. Otvor [colab.research.google.com](https://colab.research.google.com) → *File → Open notebook → GitHub*, zadaj `alzbetajaksova-ux/NLP_food_hazard`, vyber vetvu `etapa2-priprava` a súbor `notebooks/etapa2_transformer_colab.ipynb`.
2. *Runtime → Change runtime type → T4 GPU* (alebo iné ponúkané GPU) → *Save*.
3. Spusti bunky 1 a 2. Bunka 2 musí vypísať `GPU: True` a názov karty.
4. Spusti bunku 3b a povoľ prístup k Google Drive. Výsledky sa po každom behu kopírujú do `MyDrive/nlp_food_hazard_etapa2`.
5. Spusti bunku 4 (skúška na 200 správach): overí, že všetko beží; skóre budú pri takej malej vzorke takmer nulové.
6. Spusti bunku 5 (E1 a E2) a potom bunku 9 (E1b). Nechaj kartu otvorenú. Keď sa Colab odpojí, otvor notebook znova a spusti bunky 1, 2, 3, 3b a tú, ktorá bežala: hotové behy sa preskočia a pokračuje sa ďalším. Nameraný čas na Tesla T4 (15. a 16. 11.): jeden beh E1 alebo E2 (4 štítky, 4 epochy) asi 22 minút, jeden beh E1b (2 štítky, 10 epoch) asi 25 minút.
7. Bunky 6 a 7 porovnajú výsledky na valid, bunka 8 ich zabalí na stiahnutie.

## Možnosť 2: Kaggle (záloha, keď Colab nedá GPU)

Čo treba vedieť:
- Pri prvom použití GPU treba **overiť telefónne číslo** (odkaz v ľavom paneli notebooku).
- Relácia beží najviac 12 hodín. Týždenná kvóta GPU je vidieť v nastaveniach účtu (používatelia uvádzajú okolo 30 hodín, Kaggle číslo v dokumentácii neuvádza).
- Typicky ponúka T4 alebo P100.

Postup:
1. [kaggle.com](https://www.kaggle.com) → *Create → New Notebook*.
2. V pravom paneli *Settings*: *Accelerator* → GPU, *Internet* → zapnuté (inak sa nestiahne model ani repozitár).
3. Do prvej bunky:
   ```
   !git clone -b etapa2-priprava https://github.com/alzbetajaksova-ux/NLP_food_hazard.git
   %cd NLP_food_hazard
   !pip install -q transformers==5.23.0
   ```
4. Ďalej rovnaké príkazy ako v Colab notebooku (bunky 3 až 7; bunka 3b sa na Kaggle preskočí).
5. Výsledky sú v `/kaggle/working/NLP_food_hazard/results/etapa2/`; stiahni ich cez *Output* alebo príkazom `!zip -qr vysledky.zip results/etapa2` a stiahnutím súboru.

## Keď niečo nejde

- **Chyba nedostatku pamäte GPU (CUDA out of memory):** zníž `--batch` na 8, prípadne `--max-dlzka` na 192, a zapíš to do správy.
- **Málo času:** najprv dokonči E2 so semienkom 42 pre všetky štítky (to je najdôležitejší beh), potom ostatné semienka.
- **Model sa nestiahne:** skontroluj internet (na Kaggle prepínač *Internet*).
- **Výsledky sa nezhodujú medzi behmi s rovnakým semienkom:** malé rozdiely na GPU sú možné; preto používame 3 semienka a uvádzame priemer a rozptyl.

Do správy zapíš: platformu, typ GPU, čas behov (je v `results/etapa2/logy/*.json`) a všetky zmeny nastavení oproti predvoleným.

Zdroje: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [Neuromatch: Using Kaggle](https://precourse.neuromatch.io/tutorials/TechnicalHelp/Tutorial_kaggle.html), [Free Cloud GPUs in 2026](https://gpuperhour.com/blog/free-cloud-gpus-and-credits).
