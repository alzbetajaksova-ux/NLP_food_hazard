# Taxonómia chýb (2. etapa)

Taxonómia je zapísaná **pred** anotáciou chýb; dátum a obsah potvrdzuje commit v gite. Po začatí anotácie sa kategórie nemenia. Ak sa ukáže, že niečo chýba, zapíše sa to do poznámky a do správy ako obmedzenie, nie ako nová kategória.

## Čo anotujeme

- **Vzorka:** chyby najlepšieho systému na **valid** (nie na teste). Zlepšenie navrhnuté z analýzy sa potom overí na oddelených dátach: na teste (raz, na konci) a na našich nových dátach (E5).
- **Jednotka:** jedna chybná predikcia jedného štítku (`hazard` alebo `product`) v jednej správe. Ak sa v správe mýli oboje, sú to dve položky.
- **Veľkosť:** približne 200 položiek, náhodne vybraných so semienkom 42 skriptom `src/chyby_vzorka.py` (pomer hazard : produkt podľa počtu chýb, najmenej 60 položiek pre hazard).
- **Anotátori:** osoba 2 a osoba 3, každá samostatne vo svojom súbore, bez toho, aby videla odpoveď druhej. Autor taxonómie (osoba 1) neanotuje, len rozhoduje nezhody.

## Kategórie

Každej položke sa priradí **práve jedna** kategória. Ak sedí viac, použije sa tá, ktorá je v tabuľke vyššie (poradie = priorita).

| Kód | Kategória | Kedy ju použiť | Príklad z dát |
| --- | --- | --- | --- |
| A | Chyba anotácie | Pravdivé označenie zjavne nezodpovedá textu správy, predikcia textu zodpovedá lepšie. | test 362: titulok o E. coli O121, pravda „cashew“ (allergens) |
| K | Konvencia anotácie | Text pripúšťa predikciu, ale dataset v podobných prípadoch používa iné označenie (napr. hranica fraud a allergens pri nesprávnom označení). | test 893: „undeclared gluten“ v produkte „Gluten Free“, pravda fraud |
| G | Iná granularita | Predikcia je nadradená, podradená alebo takmer rovnaká trieda (napr. „chicken“ namiesto „chicken breast“). | valid 31: pravda „chicken breast“, predikcia „chicken“ |
| P | Prekrývajúce sa triedy | Produkt alebo riziko obsahovo patrí do viacerých tried a obe sú rozumné, nejde o nad- a podtriedu. | test 438: jablkový džús, pravda fruits and vegetables, predikcia nápoj |
| Z | Zavádzajúce slovo | Model zjavne reagoval na slovo alebo frázu v texte, ktorá nie je dôvodom stiahnutia alebo nie je produktom. | test 250: „Fruit Chaat Seasoning“ → fruits; test 746: „swollen cans“ → packaging defect |
| R | Vzácna alebo nevidená trieda | Pravdivá trieda má v train menej než 5 príkladov alebo v train vôbec nie je, a žiadna z kategórií vyššie nesedí. | test 963: hazard „migration“ (3 príklady v train) |
| N | Neinformatívny text | Titulok ani text nenesú dosť informácie na určenie triedy (šablóna, orezaný text, len názov firmy). | |
| I | Iné | Nič z uvedeného. Povinná poznámka, prečo. | |

Skript pre každú položku pridá pomocné stĺpce (počet príkladov pravdivej triedy v train, či je predikcia podreťazcom pravdy alebo naopak). Slúžia len na rýchlejšiu orientáciu; o kategórii rozhoduje anotátor po prečítaní titulku a textu.

## Postup

1. Osoba 1 spustí `python src/chyby_vzorka.py --system <najlepší systém>`; vzniknú `results/etapa2/chyby/anotacia_A.csv`, `anotacia_B.csv` a `kluc.csv`. Vzorka bola vytvorená 17. 11. 2026 zo systému `E12kens` (60 chýb hazardu a 140 chýb produktu z 70 a 258 chýb na valid).
2. Osoba 2 vyplní `anotacia_A.csv`, osoba 3 `anotacia_B.csv` (stĺpce `kategoria` = kód z tabuľky, `poznamka`). Navzájom si súbory neukazujú, kým nie sú obe hotové.
3. Osoba 1 spustí `python src/zhoda.py`: Cohenova kappa, zhoda po kategóriách, zoznam nezhôd.
4. Nezhody osoba 1 rozhodne a zapíše do `results/etapa2/chyby/rozhodnute.csv` s krátkym zdôvodnením.
5. Z výsledného rozdelenia sa navrhne cielená úprava (napr. pre kategóriu G alebo K) a jej prínos sa overí novým experimentom na oddelených dátach.

## Použitie LLM

Kategórie priraďujú ľudia. LLM sa na predvyplnenie nepoužíva; zadanie poloautomatickú anotáciu bez dôslednej ľudskej kontroly nepovažuje za bonusový prínos.
