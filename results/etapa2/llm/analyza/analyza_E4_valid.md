# Analýza E4 na valid: kde LLM pomáha a kde škodí

Základ: E12kens (kandidát č. 1). Vytvorené skriptom `src/e4_analyza.py`; zmena = výsledok E4 iný než kandidát 1. Zápis „+a / −b“: a správ opravil (pomohol), b pokazil (uškodil).

Systémy: `E4_Qwen253BInstruct_E12kens_f030_prod` = 3B spojenie, product (hlavný); `E4_Qwen253BInstruct_E12kens_f010` = 3B spojenie, oba; `E4_Qwen253BInstruct_E12kens` = 3B sám; `E4_Qwen257BInstructbnb4_E12kens_f020_prod` = 7B spojenie, product; `E4_Qwen257BInstructbnb4_E12kens` = 7B sám

## 1 Prehľad zmien

| systém | štítok | zmenené | pomohol | uškodil | zlé → zlé | čistý zisk | presnosť základ % | presnosť E4 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3B spojenie, product (hlavný) | hazard | 0 | 0 | 0 | 0 | 0 | 87.6 | 87.6 |
| 3B spojenie, product (hlavný) | product | 126 | 38 | 18 | 70 | 20 | 54.3 | 57.9 |
| 3B spojenie, oba | hazard | 14 | 7 | 6 | 1 | 1 | 87.6 | 87.8 |
| 3B spojenie, oba | product | 47 | 19 | 7 | 21 | 12 | 54.3 | 56.5 |
| 3B sám | hazard | 191 | 11 | 144 | 36 | -133 | 87.6 | 64.1 |
| 3B sám | product | 302 | 47 | 113 | 142 | -66 | 54.3 | 42.7 |
| 7B spojenie, product | hazard | 0 | 0 | 0 | 0 | 0 | 87.6 | 87.6 |
| 7B spojenie, product | product | 95 | 32 | 13 | 50 | 19 | 54.3 | 57.7 |
| 7B sám | hazard | 211 | 16 | 164 | 31 | -148 | 87.6 | 61.4 |
| 7B sám | product | 241 | 49 | 84 | 108 | -35 | 54.3 | 48.1 |

## 2 hazard: podľa poradie správnej triedy

| poradie správnej triedy | n | presnosť základu | 3B spojenie, oba | 3B sám | 7B sám |
| --- | --- | --- | --- | --- | --- |
| 1 | 495 | 100.0 | +0 / −6 | +0 / −144 | +0 / −164 |
| 2 | 24 | 0.0 | +6 / −0 | +8 / −0 | +7 / −0 |
| 3–5 | 21 | 0.0 | +1 / −0 | +1 / −0 | +6 / −0 |
| 6–10 | 8 | 0.0 | +0 / −0 | +2 / −0 | +3 / −0 |
| mimo top-10 | 17 | 0.0 | +0 / −0 | +0 / −0 | +0 / −0 |

## 2 product: podľa poradie správnej triedy

| poradie správnej triedy | n | presnosť základu | 3B spojenie, product (hlavný) | 3B spojenie, oba | 3B sám | 7B spojenie, product | 7B sám |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 307 | 100.0 | +0 / −18 | +0 / −7 | +0 / −113 | +0 / −13 | +0 / −84 |
| 2 | 62 | 0.0 | +29 / −0 | +18 / −0 | +27 / −0 | +29 / −0 | +33 / −0 |
| 3–5 | 43 | 0.0 | +7 / −0 | +1 / −0 | +13 / −0 | +3 / −0 | +13 / −0 |
| 6–10 | 18 | 0.0 | +2 / −0 | +0 / −0 | +7 / −0 | +0 / −0 | +3 / −0 |
| mimo top-10 | 135 | 0.0 | +0 / −0 | +0 / −0 | +0 / −0 | +0 / −0 | +0 / −0 |

## 3 hazard: podľa istota klasifikátora (p kandidáta 1)

| istota klasifikátora (p kandidáta 1) | n | presnosť základu | 3B spojenie, oba | 3B sám | 7B sám |
| --- | --- | --- | --- | --- | --- |
| < 0,3 | 4 | 25.0 | +1 / −1 | +1 / −1 | +2 / −0 |
| 0,3–0,5 | 13 | 30.8 | +2 / −2 | +1 / −4 | +4 / −4 |
| 0,5–0,7 | 37 | 40.5 | +4 / −3 | +4 / −11 | +6 / −11 |
| 0,7–0,9 | 25 | 68.0 | +0 / −0 | +2 / −7 | +3 / −9 |
| ≥ 0,9 | 486 | 94.2 | +0 / −0 | +3 / −121 | +1 / −140 |

## 3 product: podľa istota klasifikátora (p kandidáta 1)

| istota klasifikátora (p kandidáta 1) | n | presnosť základu | 3B spojenie, product (hlavný) | 3B spojenie, oba | 3B sám | 7B spojenie, product | 7B sám |
| --- | --- | --- | --- | --- | --- | --- | --- |
| < 0,3 | 20 | 5.0 | +2 / −0 | +2 / −0 | +1 / −0 | +3 / −1 | +3 / −1 |
| 0,3–0,5 | 93 | 29.0 | +13 / −5 | +6 / −4 | +14 / −11 | +10 / −7 | +16 / −9 |
| 0,5–0,7 | 113 | 30.1 | +17 / −6 | +11 / −3 | +19 / −12 | +16 / −3 | +19 / −6 |
| 0,7–0,9 | 99 | 51.5 | +5 / −7 | +0 / −0 | +9 / −28 | +3 / −2 | +7 / −20 |
| ≥ 0,9 | 240 | 80.8 | +1 / −0 | +0 / −0 | +4 / −62 | +0 / −0 | +4 / −48 |

## 4 hazard: podľa počet správ s triedou v train

| počet správ s triedou v train | n | presnosť základu | 3B spojenie, oba | 3B sám | 7B sám |
| --- | --- | --- | --- | --- | --- |
| 1–5 | 11 | 54.5 | +1 / −1 | +1 / −1 | +1 / −0 |
| 6–20 | 54 | 64.8 | +3 / −0 | +2 / −9 | +6 / −7 |
| 21–100 | 114 | 78.1 | +1 / −3 | +4 / −24 | +5 / −26 |
| > 100 | 386 | 94.6 | +2 / −2 | +4 / −110 | +4 / −131 |

## 4 product: podľa počet správ s triedou v train

| počet správ s triedou v train | n | presnosť základu | 3B spojenie, product (hlavný) | 3B spojenie, oba | 3B sám | 7B spojenie, product | 7B sám |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (nie je v train) | 49 | 0.0 | +0 / −0 | +0 / −0 | +0 / −0 | +0 / −0 | +0 / −0 |
| 1–5 | 143 | 44.1 | +13 / −6 | +7 / −4 | +19 / −23 | +11 / −4 | +16 / −12 |
| 6–20 | 148 | 56.1 | +11 / −6 | +3 / −1 | +12 / −22 | +8 / −4 | +15 / −22 |
| 21–100 | 190 | 68.4 | +14 / −6 | +9 / −2 | +16 / −54 | +13 / −5 | +18 / −44 |
| > 100 | 35 | 88.6 | +0 / −0 | +0 / −0 | +0 / −14 | +0 / −0 | +0 / −6 |

## 5 hazard: podľa kategória správnej triedy

| kategória správnej triedy | n | presnosť základu | 3B spojenie, oba | 3B sám | 7B sám |
| --- | --- | --- | --- | --- | --- |
| allergens | 207 | 89.9 | +2 / −0 | +4 / −103 | +2 / −133 |
| biological | 194 | 95.9 | +3 / −1 | +3 / −12 | +3 / −3 |
| foreign bodies | 63 | 88.9 | +0 / −2 | +2 / −8 | +4 / −9 |
| fraud | 41 | 61.0 | +0 / −1 | +0 / −10 | +5 / −8 |
| chemical | 28 | 89.3 | +1 / −0 | +1 / −6 | +1 / −5 |
| other hazard | 14 | 57.1 | +0 / −2 | +0 / −3 | +1 / −4 |
| organoleptic aspects | 8 | 62.5 | +0 / −0 | +0 / −0 | +0 / −0 |
| packaging defect | 8 | 25.0 | +1 / −0 | +1 / −1 | +0 / −1 |
| food additives and flavourings | 2 | 100.0 | +0 / −0 | +0 / −1 | +0 / −1 |

## 5 product: podľa kategória správnej triedy

| kategória správnej triedy | n | presnosť základu | 3B spojenie, product (hlavný) | 3B spojenie, oba | 3B sám | 7B spojenie, product | 7B sám |
| --- | --- | --- | --- | --- | --- | --- | --- |
| meat, egg and dairy products | 146 | 43.8 | +14 / −4 | +7 / −1 | +15 / −26 | +11 / −2 | +15 / −16 |
| cereals and bakery products | 75 | 65.3 | +3 / −3 | +1 / −1 | +4 / −20 | +3 / −2 | +6 / −16 |
| prepared dishes and snacks | 56 | 48.2 | +4 / −3 | +3 / −1 | +5 / −13 | +5 / −1 | +6 / −9 |
| fruits and vegetables | 52 | 61.5 | +2 / −1 | +0 / −0 | +3 / −8 | +1 / −1 | +2 / −6 |
| soups, broths, sauces and condiments | 36 | 50.0 | +1 / −0 | +0 / −0 | +3 / −7 | +1 / −0 | +3 / −1 |
| nuts, nut products and seeds | 33 | 60.6 | +1 / −2 | +1 / −2 | +1 / −8 | +1 / −3 | +1 / −10 |
| seafood | 27 | 74.1 | +0 / −1 | +0 / −0 | +0 / −9 | +0 / −0 | +1 / −8 |
| confectionery | 26 | 42.3 | +5 / −1 | +2 / −1 | +5 / −2 | +2 / −0 | +5 / −1 |
| ices and desserts | 24 | 83.3 | +0 / −0 | +0 / −0 | +1 / −5 | +0 / −0 | +1 / −2 |
| herbs and spices | 19 | 42.1 | +2 / −1 | +1 / −0 | +2 / −5 | +1 / −2 | +1 / −5 |
| non-alcoholic beverages | 19 | 47.4 | +1 / −0 | +1 / −0 | +2 / −2 | +1 / −0 | +1 / −0 |
| cocoa and cocoa preparations, coffee and tea | 15 | 53.3 | +1 / −1 | +1 / −0 | +1 / −3 | +2 / −1 | +2 / −4 |
| dietetic foods, food supplements, fortified foods | 14 | 64.3 | +1 / −1 | +1 / −1 | +1 / −3 | +1 / −1 | +1 / −4 |
| other food product / mixed | 10 | 30.0 | +2 / −0 | +0 / −0 | +2 / −1 | +1 / −0 | +1 / −1 |
| alcoholic beverages | 7 | 71.4 | +1 / −0 | +1 / −0 | +2 / −0 | +1 / −0 | +2 / −0 |
| fats and oils | 4 | 75.0 | +0 / −0 | +0 / −0 | +0 / −1 | +0 / −0 | +0 / −1 |
| feed materials | 1 | 0.0 | +0 / −0 | +0 / −0 | +0 / −0 | +1 / −0 | +1 / −0 |
| pet feed | 1 | 100.0 | +0 / −0 | +0 / −0 | +0 / −0 | +0 / −0 | +0 / −0 |

## 6 Macro-F1 ako v ST2: 3B spojenie, product (hlavný)

Product sa hodnotí len na správach so správnym hazardom. Macro-F1 sa priemeruje cez zjednotenie skutočných a predikovaných tried, takže predikovaná trieda, ktorá v hodnotenej časti nie je, pridá do priemeru nulu.

| štítok | macro-F1 základ | macro-F1 E4 | tried v priemere: základ | tried v priemere: E4 | predikované triedy, ktoré v (hodnotenej časti) valid nie sú: základ | …: E4 |
| --- | --- | --- | --- | --- | --- | --- |
| hazard | 0.6545 | 0.6545 | 95 | 95 | 2 | 2 |
| product | 0.3353 | 0.3796 | 362 | 358 | 76 | 72 |

## 6 product: triedy s najväčšou stratou a ziskom F1

| trieda | správ v valid | F1 základ | F1 E4 | zmena F1 | príspevok k macro-F1 (≈ zmena / počet tried) | len predikovaná (nie je v valid) |
| --- | --- | --- | --- | --- | --- | --- |
| chilled salads | 1 | 1.0 | 0.0 | -1.0 | -0.0028 | False |
| dried beef meat | 1 | 1.0 | 0.0 | -1.0 | -0.0028 | False |
| energy and protein bars | 1 | 1.0 | 0.0 | -1.0 | -0.0028 | False |
| milk chocolates | 1 | 1.0 | 0.0 | -1.0 | -0.0028 | False |
| mixed vegetables | 1 | 1.0 | 0.0 | -1.0 | -0.0028 | False |
| pancakes | 1 | 0.667 | 0.0 | -0.667 | -0.0019 | False |
| dried fruits | 2 | 1.0 | 0.5 | -0.5 | -0.0014 | False |
| thermal processed pork meat | 1 | 0.5 | 0.0 | -0.5 | -0.0014 | False |
| dried mushrooms | 1 | 0.0 | 1.0 | 1.0 | 0.0028 | False |
| ginger slices | 1 | 0.0 | 1.0 | 1.0 | 0.0028 | False |
| goat cheese | 1 | 0.0 | 1.0 | 1.0 | 0.0028 | False |
| ground almonds | 1 | 0.0 | 1.0 | 1.0 | 0.0028 | False |
| mung bean sprouts | 2 | 0.0 | 1.0 | 1.0 | 0.0028 | False |
| pork belly | 1 | 0.0 | 1.0 | 1.0 | 0.0028 | False |
| precooked cooked beef meat products | 1 | 0.0 | 1.0 | 1.0 | 0.0028 | False |
| rice cakes | 1 | 0.0 | 1.0 | 1.0 | 0.0028 | False |

## 7 3B spojenie, product (hlavný), product: najčastejšie zámeny, keď uškodil

| kandidát 1 (základ) | výsledok E4 | počet |
| --- | --- | --- |
| Catfishes (freshwater) | frozen fish products | 1 |
| biscuits | cookies | 1 |
| cheese | unpasteurised cheese | 1 |
| chilled salads | salads | 1 |
| corn | pop corn | 1 |
| dried beef meat | jerky products | 1 |
| energy and protein bars | protein balls | 1 |
| hazelnuts | chocolate and hazelnut spread | 1 |
| marshmalows | candies | 1 |
| milk chocolates | chocolate | 1 |

## 7 3B spojenie, product (hlavný), product: najčastejšie zámeny, keď pomohol

| kandidát 1 (základ) | výsledok E4 | počet |
| --- | --- | --- |
| bean sprouts | mung bean sprouts | 1 |
| beef | precooked cooked beef meat products | 1 |
| bread | bread mix | 1 |
| cakes | brownies | 1 |
| cereal bars | chocolate bars | 1 |
| chicken preparations | chicken products - frozen | 1 |
| coffee beans | coffee drink | 1 |
| cookies | snacks (various) | 1 |
| cow milk cheese | cheese | 1 |
| curry | ready to eat - cook meals | 1 |

## 7 3B sám, hazard: najčastejšie zámeny, keď uškodil

| kandidát 1 (základ) | výsledok E4 | počet |
| --- | --- | --- |
| milk and products thereof | undeclared constituent | 24 |
| cereals containing gluten and products thereof | allergens | 12 |
| soybeans and products thereof | undeclared constituent | 10 |
| eggs and products thereof | undeclared constituent | 9 |
| milk and products thereof | allergens | 8 |
| listeria monocytogenes | microbiological contamination | 4 |
| eggs and products thereof | allergens | 4 |
| peanuts and products thereof | allergens | 4 |
| peanuts and products thereof | undeclared constituent | 3 |
| soybeans and products thereof | allergens | 3 |

## 7 3B sám, hazard: najčastejšie zámeny, keď pomohol

| kandidát 1 (základ) | výsledok E4 | počet |
| --- | --- | --- |
| cashew | hazelnut | 1 |
| eggs and products thereof | sulphur dioxide and sulphites | 1 |
| escherichia coli | metal fragment | 1 |
| incorrect labeling | peanuts and products thereof | 1 |
| listeria spp | escherichia coli | 1 |
| packaging defect | bulging packaging | 1 |
| packaging defect | yeasts | 1 |
| plastic fragment | foreign bodies | 1 |
| salmonella | cronobacter spp | 1 |
| salmonella | other | 1 |

## 7 3B sám, product: najčastejšie zámeny, keď uškodil

| kandidát 1 (základ) | výsledok E4 | počet |
| --- | --- | --- |
| corn | pop corn | 3 |
| chicken based products | chicken preparations | 3 |
| soup | soups | 3 |
| Fishes not identified | structured/textured fish meat products or fish paste | 2 |
| biscuits | cookies | 2 |
| peanuts | unsalted organic crunchy peanut butter | 2 |
| ice cream | ices and desserts | 2 |
| salads | prepared salads | 2 |
| seasoning | seasoning mix | 2 |
| chicken based products | chicken products - frozen | 2 |

## 7 3B sám, product: najčastejšie zámeny, keď pomohol

| kandidát 1 (základ) | výsledok E4 | počet |
| --- | --- | --- |
| bean sprouts | mung bean sprouts | 2 |
| chicken based products | chicken breast | 2 |
| Veggie Burger | frozen burgers | 1 |
| bread | bread mix | 1 |
| bottled mineral water | bottled water | 1 |
| cakes | brownies | 1 |
| chicken preparations | chicken products - frozen | 1 |
| coffee beans | coffee drink | 1 |
| cow milk cheese | cheese | 1 |
| curry | ready to eat - cook meals | 1 |

## 8 Príklady: 3B spojenie, product (hlavný) (náhodne, seed 42)

| štítok | LLM | id | titulok | správna | kandidát 1 | E4 | poradie správnej | p klasifikátora (kand. 1) | p LLM (jeho výber) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| product | pomohol | 102 | Mikes Meats Pty Ltd—Prager Leg Ham | ham | sliced ham | ham | 2 | 0.545 | 1.0 |
| product | pomohol | 124 | 2009 - zachary confections, inc. announces a nationwide voluntary recall of market pantry chocolate covered pe | candies | peanuts | candies | 2 | 0.705 | 1.0 |
| product | pomohol | 271 | The Great Northern Sandwich Co is recalling its Spar Daily Deli Peppered Ham & Egg Sandwiches because they con | sandwiches | ham | sandwiches | 2 | 0.563 | 1.0 |
| product | pomohol | 282 | Tiny Rebel Brewing Co. recalls bottles of Stay Puft Marshmallow Porter because lactose is not declared as bein | beer | marshmalows | beer | 2 | 0.618 | 1.0 |
| product | pomohol | 301 | Bhugga sold at Rajthani Sweets & Restaurants recalled due to Salmonella | sweets | dietary supplement | sweets | 2 | 0.317 | 1.0 |
| product | pomohol | 397 | Noshu Foods — Noshu Sugar Free Fudgy Brownies Mix 300g | brownies | cakes | brownies | 2 | 0.93 | 1.0 |
| product | uškodil | 84 | New Jersey Firm Recalls Cobb and Spinach Salads Due to Possible Listeria Contamination | chilled salads | chilled salads | salads | 1 | 0.691 | 0.681 |
| product | uškodil | 307 | IKEA recalls its GODIS PÅSKKYCKLING Marshmallow candy due to possible mice infestation | marshmalows | marshmalows | candies | 1 | 0.508 | 1.0 |
| product | uškodil | 324 | Venda Ravioli, Inc. Recalls Meat Ravioli Products Produced Without Benefit of Inspection | ravioli | ravioli | meat and meat products (other than poultry) | 1 | 0.786 | 1.0 |
| product | uškodil | 364 | Concord Fresh Success Recall Mild Salsa Seasoning Mix Due To Undeclared Milk Allergen | seasoning | seasoning | seasoning mix | 1 | 0.748 | 1.0 |
| product | uškodil | 400 | Hy-Vee Fresh Commissary Recalls Ready-To-Eat Beef and Chicken Products Due to Misbranding and Undeclared Aller | thermal processed beef meat | thermal processed beef meat | ready to eat - cook meals | 1 | 0.377 | 0.855 |
| product | uškodil | 529 | Godiva Australia — Chocolate Domes - Milk Chocolate - Coconut Crunch 124g | milk chocolates | milk chocolates | chocolate | 1 | 0.625 | 0.964 |
