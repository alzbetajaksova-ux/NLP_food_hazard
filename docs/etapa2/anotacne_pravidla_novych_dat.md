# Anotačné pravidlá pre nové dáta (2. etapa, bonus: vlastný dataset)

Cieľ: malý vlastný testovací súbor **novších** správ o stiahnutí potravín (dataset úlohy končí rokom 2022), anotovaný rovnakou schémou ako SemEval-2025 Task 9. Slúži na overenie, ako systém funguje na nových dátach (E5), a ako oddelené dáta pre potvrdenie zlepšenia z analýzy chýb.

Pravidlá sú zapísané **pred** začatím anotácie. Zmena počas anotácie sa zapíše do zoznamu zmien na konci súboru aj s dátumom.

## Zber (osoba 3)

- **Zdroje:** verejné stránky úradov, ktoré zverejňujú oznámenia o stiahnutí potravín. [doplniť konkrétne zdroje a overiť ich podmienky použitia a licenciu pred zberom]
- **Obdobie:** správy zverejnené od roku 2023. [doplniť presné obdobie]
- **Rozsah:** približne 300 správ. [tím potvrdí]
- **Polia:** `id`, `zdroj_url`, `datum`, `krajina`, `title`, `text` (rovnaký význam ako v datasete úlohy).
- **Čistenie:** odstrániť duplicity, prázdne texty a správy, ktoré nie sú o potravinách; zapísať, koľko správ a prečo bolo vyradených.

## Štítky

Rovnaké štyri ako v úlohe: `hazard-category`, `product-category`, `hazard`, `product`.

1. **hazard:** konkrétna príčina stiahnutia (napr. „salmonella“, „milk and products thereof“, „plastic fragment“). Použi názov zo zoznamu hazardov v `data/incidents_train.csv`, ak niektorý sedí. Ak nesedí žiadny, napíš nový názov malými písmenami v angličtine a označ stĺpec `nova_trieda` = 1.
2. **hazard-category:** podľa toho, do ktorej kategórie patrí zvolený hazard v train (každý hazard tam patrí do jedinej kategórie). Pri novom hazarde vyber z 10 existujúcich kategórií.
3. **product:** konkrétny stiahnutý produkt čo najbližšie k triedam v train. Ak sedí trieda z train, použi ju presne (aj s veľkosťou písmen). Inak nový názov malými písmenami, jednotné číslo, bez značky a balenia, a `nova_trieda` = 1.
4. **product-category:** jedna z 22 existujúcich kategórií.

## Rozhodovacie pravidlá

- Hodnotí sa **dôvod stiahnutia**, nie všetko, čo text spomína (napr. „swollen cans“ ako prejav kontaminácie → hazard je baktéria, nie chyba obalu).
- **Nedeklarovaný alergén** (alergén v produkte, ktorý nie je na etikete): hazard = alergén, kategória allergens.
- **Nesprávne označenie bez konkrétneho alergénu** (zlý jazyk etikety, chýbajúce údaje, nepravdivé tvrdenie): kategória fraud. [doplniť príklady z train pre konzistentnosť, lebo dataset tu nie je jednotný]
- Viac produktov v jednej správe: anotuj ten, ktorý je v titulku prvý.
- Ak sa nedá rozhodnúť, vyplň `poznamka` a nechaj štítok prázdny; rozhodne osoba 1.

## Postup

- Osoba 2 a osoba 3 anotujú **každá samostatne** všetky správy (najmenej 100 spoločných, ak nebude čas na všetky) vo vlastnom súbore.
- Osoba 1 (autor pravidiel) neanotuje. Spočíta zhodu (`python src/zhoda.py` s rovnakou logikou, po štítkoch), rozhodne nezhody a zapíše dôvody.
- LLM sa na predvyplnenie štítkov nepoužíva.
- Výsledný súbor sa zverejní s licenciou kompatibilnou so zdrojmi, datasheetom (zdroje, obdobie, postup, zhoda anotátorov, známe obmedzenia) a skriptom na spracovanie. [doplniť licenciu po overení zdrojov]

## Zoznam zmien

- (zatiaľ žiadne)
