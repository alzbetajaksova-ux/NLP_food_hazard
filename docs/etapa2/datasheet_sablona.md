# Datasheet: [názov datasetu] (šablóna)

Šablóna podľa otázok z *Datasheets for Datasets* (Gebru et al., 2021, arXiv:1803.09010), skrátená na to, čo sa týka nášho malého datasetu. Vypĺňa osoba 3 (zber) a osoba 1 (anotácia a zhoda). Hranaté zátvorky nahradiť skutočnými údajmi; čo nevieme, napísať „neznáme“, nie odhad.

## 1. Motivácia
- **Na čo dataset vznikol:** overiť, ako systém pre SemEval-2025 Task 9 funguje na novších správach o stiahnutí potravín (dataset úlohy končí rokom 2022), a mať oddelené dáta na potvrdenie zlepšenia z analýzy chýb.
- **Kto ho vytvoril:** Alexandra Domonkošová, Kristína Gvozdiaková, Alžbeta Jakšová, projekt z predmetu Spracovanie prirodzeného jazyka, FEI TUKE, 2026.
- **Financovanie:** žiadne.

## 2. Zloženie
- **Čo je jedna položka:** jedna verejná správa o stiahnutí potraviny (titulok, text, dátum, krajina, zdroj) so štyrmi štítkami.
- **Počet položiek:** [počet] (zozbierané [počet], vyradené [počet] a prečo).
- **Obdobie a krajiny:** [od – do], [krajiny a počty].
- **Štítky:** `hazard-category`, `product-category`, `hazard`, `product` podľa `docs/etapa2/anotacne_pravidla_novych_dat.md`.
- **Nové triedy:** [počet hazardov a produktov, ktoré nie sú v train úlohy].
- **Rozdelenie tried:** [tabuľka alebo graf počtov kategórií].
- **Chýbajúce alebo neisté údaje:** [napr. orezané texty, správy bez dátumu].
- **Osobné alebo citlivé údaje:** [overiť; správy môžu obsahovať názvy firiem, nie údaje o osobách].

## 3. Zber
- **Zdroje:** [zoznam stránok s URL a dátumom prístupu].
- **Podmienky použitia zdrojov:** [citácia alebo odkaz na podmienky každého zdroja; čo dovoľujú].
- **Spôsob zberu:** [ručne / skriptom; ak skriptom, odkaz na skript v repozitári].
- **Výber správ:** [všetky v období / náhodný výber / prvých N; podľa čoho].
- **Kto zbieral a kedy:** [osoba, dátumy].

## 4. Spracovanie a anotácia
- **Čistenie:** [čo sa odstraňovalo: duplicity, HTML, prázdne texty, nepotravinové správy].
- **Anotačné pravidlá:** `docs/etapa2/anotacne_pravidla_novych_dat.md` (verzia z commitu [hash]).
- **Anotátori:** osoba 2 a osoba 3, každá samostatne; nezhody rozhodla osoba 1 (autorka pravidiel).
- **Zhoda anotátorov:** [Cohenova kappa po štítkoch zo `src/zhoda.py`, počet spoločne anotovaných položiek].
- **Analýza nezhôd:** [najčastejšie typy nezhôd a ako sa rozhodli].
- **Použitie LLM:** pri anotácii sa LLM nepoužilo. [ak áno, presne kde a ako sa kontrolovalo]
- **Surové dáta:** [uložené / neuložené a prečo].

## 5. Použitie
- **Na čo sa použil:** E5, overenie zlepšenia z analýzy chýb.
- **Na čo sa nehodí:** [napr. trénovanie veľkých modelov kvôli malému rozsahu; zovšeobecnenie mimo krajín zdrojov].
- **Známe skreslenia:** [napr. prevaha jednej krajiny, len anglické správy].

## 6. Distribúcia a licencia
- **Kde je zverejnený:** [repozitár / Zenodo / iné].
- **Licencia:** [licencia kompatibilná s podmienkami zdrojov; dataset úlohy je CC BY-NC-SA 4.0].
- **Obmedzenia:** [napr. nekomerčné použitie].

## 7. Údržba
- **Kto ho spravuje a kontakt:** [meno, e-mail alebo GitHub].
- **Bude sa aktualizovať:** [áno / nie].
- **Ako hlásiť chyby:** [GitHub issues v repozitári].
