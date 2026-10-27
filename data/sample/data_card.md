# Data card

Generado con `python -m src.generate_data` (semilla 1234, tamaños por nivel {'n_train': 50, 'n_val': 50, 'n_test': 50}, operando máximo 20, valor máximo 99).

Redacciones reservadas para `test_tpl` (18): `ba_tantos`, `bd_inv2`, `bh_inv`, `bm2_veces`, `bm3_k`, `bs_k`, `ca_nina`, `cd_parte`, `ch_lleva`, `cm2_tantos`, `cm3_c`, `cs_nino`, `pa_recibe`, `pd_parte`, `ph_pierde`, `pm2_sedup`, `pm3_setrip`, `ps_vende`

## Tamaños y longitudes (tokens por ejemplo, incluido `<eos>`)

| Fichero | Nivel | Problemas | Tipo A | B desordenados | Verbose | Long. media | Long. máx. | Respuesta media | Respuesta máx. |
|---|---|---|---|---|---|---|---|---|---|
| train | 1 | 50 | 50% | 40% | 84% | 161 | 218 | 15.8 | 54 |
| train | 2 | 50 | 50% | 76% | 70% | 190 | 276 | 24.1 | 84 |
| train | 3 | 50 | 50% | 48% | 52% | 221 | 316 | 27.0 | 93 |
| train | 4 | 50 | 50% | 56% | 44% | 247 | 337 | 25.2 | 98 |
| val | 1 | 50 | 50% | 52% | 72% | 153 | 209 | 18.0 | 57 |
| val | 2 | 50 | 50% | 64% | 66% | 189 | 262 | 22.5 | 99 |
| val | 3 | 50 | 50% | 36% | 54% | 222 | 315 | 25.5 | 93 |
| val | 4 | 50 | 50% | 60% | 34% | 246 | 331 | 22.9 | 86 |
| test_id | 1 | 50 | 50% | 44% | 80% | 154 | 199 | 15.3 | 60 |
| test_id | 2 | 50 | 50% | 44% | 54% | 185 | 268 | 21.0 | 81 |
| test_id | 3 | 50 | 50% | 56% | 60% | 222 | 306 | 23.5 | 92 |
| test_id | 4 | 50 | 50% | 60% | 36% | 242 | 336 | 26.3 | 84 |
| test_tpl | 1 | 50 | 50% | 36% | 72% | 155 | 203 | 13.2 | 57 |
| test_tpl | 2 | 50 | 50% | 40% | 56% | 184 | 248 | 24.3 | 87 |
| test_tpl | 3 | 50 | 50% | 60% | 48% | 222 | 309 | 25.3 | 92 |
| test_tpl | 4 | 50 | 50% | 40% | 38% | 252 | 333 | 23.4 | 99 |
| test_ood | 5 | 50 | 50% | 60% | 8% | 253 | 334 | 26.7 | 97 |
| test_ood | 6 | 50 | 50% | 32% | 0% | 276 | 334 | 31.5 | 99 |

## Operaciones en train

| Operación | Frecuencia |
|---|---|
| suma | 27.8% |
| resta | 24.8% |
| doble | 20.2% |
| mitad | 9.2% |
| triple | 9.0% |
| cuarta parte | 5.0% |
| tercera parte | 4.0% |

## Respuestas en train

- Valores distintos: 64; las 5 más frecuentes: 1 (7.5%), 6 (6.0%), 3 (5.5%), 9 (5.0%), 2 (5.0%)
- Respuestas de una cifra: 39.5%; de dos cifras: 60.5%; de tres cifras: 0.0%
- Marcos: A-persona 24%, B 50%, A-recipiente 26%

## Referencias: heurísticas que no razonan

Exactitud que se obtendría sin resolver nada. Cualquier modelo tiene que superarlas con claridad.

| Fichero | Nivel | Siempre la respuesta más frecuente de train | Primer número del texto | Último número del texto |
|---|---|---|---|---|
| test_id | 1 | 12.0 % | 0.0 % | 6.0 % |
| test_id | 2 | 14.0 % | 8.0 % | 4.0 % |
| test_id | 3 | 8.0 % | 8.0 % | 6.0 % |
| test_id | 4 | 6.0 % | 0.0 % | 8.0 % |
| test_ood | 5 | 8.0 % | 8.0 % | 4.0 % |
| test_ood | 6 | 6.0 % | 2.0 % | 2.0 % |

La respuesta más frecuente de train es 1.

## Trazas de evaluación que ya aparecen en train

Porcentaje de problemas cuya traza (mismos números y operaciones) aparece en train con otro texto. No es fuga de datos (el texto es distinto), pero en los niveles bajos la aritmética ya se ha visto.

| Fichero | Nivel 1 | Nivel 2 | Nivel 3 | Nivel 4 | Nivel 5 | Nivel 6 |
|---|---|---|---|---|---|---|
| test_id | 24.0 % | 4.0 % | 0.0 % | 0.0 % | — | — |
| test_tpl | 14.0 % | 4.0 % | 0.0 % | 0.0 % | — | — |
| test_ood | — | — | — | — | 0.0 % | 0.0 % |

## Ejemplos

- **train · nivel 3 · tipo A**: A Dani le gusta coleccionar bombones. Al principio, Dani tiene 9 bombones. Un amigo le regala 19 bombones. Entonces reparte sus bombones en 4 montones iguales y se queda con uno. Su abuelo le regala 17 bombones. ¿Cuántos bombones tiene Dani ahora? → **24**
- **train · nivel 1 · tipo B**: Varios amigos comparan sus bombones. Sabemos que Ana tiene 14 bombones. Dora tiene la mitad de bombones que Ana. ¿Cuál es el número de bombones de Dora? → **7**
- **train · nivel 2 · tipo B (desordenado)**: Edu tiene la mitad de botones que Laura. Sabemos que Edu tiene 1 botón. Luis tiene tres veces lo que tiene Laura. Entonces, ¿cuántos botones tiene Luis? → **6**
- **train · nivel 4 · tipo B (desordenado)**: Sabemos que Álvaro tiene 3 canicas. Luis tiene 20 canicas menos que Hugo. Bea tiene 19 canicas más que Celia. Álvaro tiene 20 canicas menos que Celia. Luis tiene la mitad de canicas que Bea. ¿Cuál es el número de canicas de Hugo? → **41**
- **train · nivel 1 · tipo A**: Una cesta contiene 10 caramelos. La cesta está encima de la mesa. Alguien añade 11 caramelos a la cesta. ¿Cuántos caramelos hay ahora en la cesta? → **21**
- **train · nivel 3 · tipo B (desordenado)**: Dani tiene 10 flores. Alba tiene 9 flores menos que Elena. Teo tiene el triple de flores que Alba. Elena tiene el triple de flores que Dani. ¿Cuál es el número de flores de Teo? → **63**
- **train · nivel 2 · tipo A**: En una mochila hay 13 flores. Ponen 4 flores más. Duplican las flores. ¿Cuántas flores quedan en la mochila? → **34**
- **train · nivel 4 · tipo A**: Fran guarda 3 piedras en su habitación. Le da 2 piedras a su padre. Entonces una vecina le regala 15 piedras. Después Fran reparte sus piedras en 4 montones iguales y se queda con uno. Más tarde se queda solo con la mitad de sus piedras. ¿Con cuántas piedras se queda Fran? → **2**

- **val · nivel 3 · tipo A**: En una mochila hay 14 rotuladores. Sacan la mitad de los rotuladores. Sacan 4 rotuladores. Entonces ponen 13 rotuladores más. Al final, ¿cuántos rotuladores hay en la mochila? → **16**
- **val · nivel 4 · tipo B**: Varios amigos comparan sus naranjas. Javier tiene exactamente 9 naranjas. Javier tiene la mitad de naranjas que Héctor. Sergio tiene tres veces lo que tiene Héctor. Sergio tiene 11 naranjas más que Carlos. Carlos tiene la mitad de naranjas que Andrés. Entonces, ¿cuántas naranjas tiene Andrés? → **86**
- **val · nivel 1 · tipo A**: Una mochila contiene 7 rotuladores. Nadie sabe quién ha dejado allí la mochila. El número de rotuladores de la mochila se triplica. ¿Cuántos rotuladores hay ahora en la mochila? → **21**
- **val · nivel 4 · tipo A**: Clara tiene 3 cartas. Divide sus cartas en 3 grupos iguales y guarda solo uno. Entonces gana 13 cartas más. Regala 12 cartas. Se queda solo con la mitad de sus cartas. ¿Con cuántas cartas se queda Clara? → **1**
- **val · nivel 1 · tipo B**: Raúl tiene 8 postales. Elena tiene la mitad de postales que Raúl. Entonces, ¿cuántas postales tiene Elena? → **4**
- **val · nivel 2 · tipo A**: Al principio, Dani tiene 8 figuras. Multiplica por tres sus figuras. Triplica sus figuras. ¿Con cuántas figuras se queda Dani? → **72**
- **val · nivel 2 · tipo B**: Javier tiene exactamente 12 galletas. Javier tiene la mitad de galletas que Lucía. A Rocío le faltan 9 galletas para tener tantas como Lucía. ¿Cuántas galletas tiene Rocío? → **15**
- **val · nivel 3 · tipo B (desordenado)**: Edu tiene 5 conchas. Andrés tiene el doble que Laura. Inés tiene el triple de conchas que Andrés. Laura tiene el triple de conchas que Edu. ¿Cuántas conchas tiene Inés? → **90**

- **test_id · nivel 1 · tipo A**: Carlos siempre lleva la cuenta de sus bombones. Al principio, Carlos tiene 20 bombones. Carlos multiplica por tres sus bombones. ¿Cuántos bombones tiene Carlos ahora? → **60**
- **test_id · nivel 3 · tipo A**: En una caja hay 19 botones. Ponen 20 botones más. Luego hacen 3 montones iguales y dejan solo uno. Meten 14 botones más. Al final, ¿cuántos botones hay en la caja? → **27**
- **test_id · nivel 4 · tipo B (desordenado)**: Julia tiene el doble que Ana. Ana tiene 8 lápices menos que Jorge. Jorge tiene 11 lápices. Paula tiene el triple de lápices que Héctor. Héctor tiene 3 lápices menos que Julia. ¿Cuántos lápices tiene Paula? → **9**
- **test_id · nivel 4 · tipo A**: Clara tiene 9 postales. Pierde 8 postales. Gana 10 postales más. Entonces duplica sus postales. Gana 1 postal más. ¿Cuántas postales tiene Clara al final? → **23**
- **test_id · nivel 1 · tipo B**: Fran tiene exactamente 20 canicas. A Eva le faltan 13 canicas para tener tantas como Fran. ¿Cuál es el número de canicas de Eva? → **7**
- **test_id · nivel 2 · tipo A**: Iván empieza el día con 1 libro. Triplica sus libros. Pierde 2 libros. ¿Con cuántos libros se queda Iván? → **1**
- **test_id · nivel 2 · tipo B**: Raúl tiene 9 botones. Sofía tiene 11 más que Raúl. Fran tiene el triple de botones que Sofía. ¿Cuántos botones tiene Fran? → **60**
- **test_id · nivel 3 · tipo B (desordenado)**: Lucía tiene la tercera parte de las piedras de Fran. Hugo tiene la mitad de piedras que Tania. Tania tiene 1 piedra menos que Fran. Sabemos que Hugo tiene 19 piedras. ¿Cuál es el número de piedras de Lucía? → **13**

- **test_tpl · nivel 3 · tipo A**: En una bandeja hay 11 galletas. Meten tantas galletas como ya había y luego multiplican por tres las galletas. Sacan la mitad de las galletas. ¿Cuántas galletas quedan en la bandeja? → **33**
- **test_tpl · nivel 4 · tipo B**: Paula tiene 16 naranjas. Vera tiene 5 naranjas menos que Paula. Gala tiene tantas naranjas como Vera y 11 más. Dani tiene la mitad de naranjas que Gala. Sofía tiene 9 más que Dani. Entonces, ¿cuántas naranjas tiene Sofía? → **20**
- **test_tpl · nivel 2 · tipo B (desordenado)**: Varios amigos comparan sus sellos. Hugo tiene exactamente 17 sellos. Pablo tiene el cuádruple de sellos que Sara. Hugo tiene 19 sellos menos que Pablo. ¿Cuántos sellos tiene Sara? → **9**
- **test_tpl · nivel 4 · tipo A**: A Héctor le gusta coleccionar botones. Al principio, Héctor tiene 16 botones. Multiplica por tres sus botones. Más tarde un amigo le da tantos botones como ya tenía. Le da 5 botones a su hermana. Héctor recibe 3 botones de regalo. ¿Con cuántos botones se queda Héctor? → **94**
- **test_tpl · nivel 1 · tipo A**: Una bandeja contiene 8 bombones. Nadie sabe quién ha dejado allí la bandeja. Una niña mete 7 bombones más. ¿Cuántos bombones quedan en la bandeja? → **15**
- **test_tpl · nivel 1 · tipo B**: Varios amigos comparan sus conchas. Dora tiene exactamente 12 conchas. Sergio tiene 6 menos que Dora. Entonces, ¿cuántas conchas tiene Sergio? → **6**
- **test_tpl · nivel 2 · tipo A**: Teo empieza el día con 6 flores. Triplica sus flores y vende 17 flores a un compañero. ¿Cuántas flores tiene Teo al final? → **1**
- **test_tpl · nivel 3 · tipo B**: Rocío tiene exactamente 9 cuadernos. Mario tiene el triple que Rocío. Mario tiene 7 cuadernos más que Pedro. Pedro tiene el doble de cuadernos que Ángela. Entonces, ¿cuántos cuadernos tiene Ángela? → **10**

- **test_ood · nivel 6 · tipo B**: Irene tiene 3 canicas. Fran tiene 19 más que Irene. Laura tiene 2 canicas menos que Fran. Laura tiene 2 canicas más que Ana. Ciro tiene el doble de canicas que Ana. Lola tiene 1 canica más que Ciro. Tania tiene el doble de canicas que Lola. ¿Cuántas canicas tiene Tania? → **74**
- **test_ood · nivel 6 · tipo A**: Pedro tiene 19 peonzas. Regala 7 peonzas. Regala la mitad de sus peonzas. Entonces pierde 5 peonzas. Luego gana 19 peonzas más. Divide sus peonzas en 4 grupos iguales y guarda solo uno. Gana 19 peonzas más. ¿Cuántas peonzas tiene Pedro ahora? → **24**
- **test_ood · nivel 5 · tipo A**: Tania empieza el día con 15 galletas. Compra 6 galletas. Luego pierde 8 galletas. Regala 3 galletas. Pierde 6 galletas y se queda solo con la mitad de sus galletas. ¿Cuántas galletas tiene Tania al final? → **2**
- **test_ood · nivel 5 · tipo B (desordenado)**: Eva tiene 1 cuaderno menos que Iván. Celia tiene 1 más que Fran. Lola tiene la mitad de cuadernos que Celia. Eva tiene el cuádruple que Pedro. Iván tiene 9 cuadernos. Fran tiene la mitad de cuadernos que Pedro. Entonces, ¿cuántos cuadernos tiene Lola? → **1**

