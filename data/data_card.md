# Data card

Generado con `python -m src.generate_data` (semilla 1234, tamaños por nivel {'n_train': 50000, 'n_val': 1000, 'n_test': 2000}, operando máximo 20, valor máximo 99).

Redacciones reservadas para `test_tpl` (18): `ba_tantos`, `bd_inv2`, `bh_inv`, `bm2_veces`, `bm3_k`, `bs_k`, `ca_nina`, `cd_parte`, `ch_lleva`, `cm2_tantos`, `cm3_c`, `cs_nino`, `pa_recibe`, `pd_parte`, `ph_pierde`, `pm2_sedup`, `pm3_setrip`, `ps_vende`

## Tamaños y longitudes (tokens por ejemplo, incluido `<eos>`)

| Fichero | Nivel | Problemas | Tipo A | B desordenados | Verbose | Long. media | Long. máx. | Respuesta media | Respuesta máx. |
|---|---|---|---|---|---|---|---|---|---|
| train | 1 | 50000 | 50% | 50% | 80% | 154 | 241 | 14.7 | 60 |
| train | 2 | 50000 | 50% | 50% | 65% | 190 | 301 | 21.2 | 99 |
| train | 3 | 50000 | 50% | 50% | 50% | 218 | 355 | 24.2 | 99 |
| train | 4 | 50000 | 50% | 50% | 35% | 244 | 411 | 26.9 | 99 |
| val | 1 | 1000 | 50% | 49% | 81% | 154 | 228 | 14.9 | 60 |
| val | 2 | 1000 | 50% | 49% | 66% | 189 | 293 | 21.2 | 96 |
| val | 3 | 1000 | 50% | 52% | 52% | 220 | 330 | 25.5 | 99 |
| val | 4 | 1000 | 50% | 50% | 37% | 245 | 393 | 26.8 | 99 |
| test_id | 1 | 2000 | 50% | 50% | 80% | 153 | 233 | 14.7 | 60 |
| test_id | 2 | 2000 | 50% | 52% | 66% | 190 | 280 | 21.1 | 99 |
| test_id | 3 | 2000 | 50% | 52% | 53% | 219 | 330 | 24.8 | 99 |
| test_id | 4 | 2000 | 50% | 52% | 34% | 243 | 385 | 26.9 | 99 |
| test_tpl | 1 | 2000 | 50% | 49% | 80% | 152 | 218 | 14.6 | 60 |
| test_tpl | 2 | 2000 | 50% | 50% | 65% | 191 | 281 | 21.0 | 99 |
| test_tpl | 3 | 2000 | 50% | 49% | 50% | 220 | 340 | 24.0 | 99 |
| test_tpl | 4 | 2000 | 50% | 50% | 37% | 248 | 391 | 26.7 | 99 |
| test_ood | 5 | 2000 | 50% | 54% | 14% | 263 | 408 | 29.5 | 99 |
| test_ood | 6 | 2000 | 50% | 50% | 9% | 289 | 411 | 30.2 | 99 |

## Operaciones en train

| Operación | Frecuencia |
|---|---|
| suma | 27.9% |
| resta | 25.2% |
| doble | 17.8% |
| triple | 11.5% |
| mitad | 9.7% |
| tercera parte | 4.4% |
| cuarta parte | 3.5% |

## Respuestas en train

- Valores distintos: 99; las 5 más frecuentes: 1 (7.4%), 2 (6.9%), 3 (5.3%), 6 (4.3%), 4 (4.3%)
- Respuestas de una cifra: 39.4%; de dos cifras: 60.6%; de tres cifras: 0.0%
- Marcos: B 50%, A-recipiente 25%, A-persona 25%

## Referencias: heurísticas que no razonan

Exactitud que se obtendría sin resolver nada. Cualquier modelo tiene que superarlas con claridad.

| Fichero | Nivel | Siempre la respuesta más frecuente de train | Primer número del texto | Último número del texto |
|---|---|---|---|---|
| test_id | 1 | 7.6 % | 0.6 % | 3.2 % |
| test_id | 2 | 7.5 % | 3.8 % | 5.2 % |
| test_id | 3 | 6.2 % | 3.9 % | 5.5 % |
| test_id | 4 | 5.4 % | 3.0 % | 4.8 % |
| test_ood | 5 | 4.7 % | 3.2 % | 3.8 % |
| test_ood | 6 | 5.5 % | 2.6 % | 4.3 % |

La respuesta más frecuente de train es 1.

## Trazas de evaluación que ya aparecen en train

Porcentaje de problemas cuya traza (mismos números y operaciones) aparece en train con otro texto. No es fuga de datos (el texto es distinto), pero en los niveles bajos la aritmética ya se ha visto.

| Fichero | Nivel 1 | Nivel 2 | Nivel 3 | Nivel 4 | Nivel 5 | Nivel 6 |
|---|---|---|---|---|---|---|
| test_id | 100.0 % | 100.0 % | 47.7 % | 10.1 % | — | — |
| test_tpl | 100.0 % | 99.9 % | 48.5 % | 10.5 % | — | — |
| test_ood | — | — | — | — | 0.0 % | 0.0 % |

## Ejemplos

- **train · nivel 1 · tipo B (desordenado)**: Varios amigos comparan sus libros. Jorge tiene 19 libros menos que Carlos. Jorge tiene exactamente 5 libros. ¿Cuál es el número de libros de Carlos? → **24**
- **train · nivel 3 · tipo B**: Unos amigos cuentan las cartas que tiene cada uno. Hugo tiene exactamente 1 carta. Hugo tiene 13 cartas menos que Pablo. Tania tiene la mitad de cartas que Pablo. Tania tiene 14 cartas menos que Jorge. ¿Cuál es el número de cartas de Jorge? → **21**
- **train · nivel 4 · tipo A**: En una caja hay 3 globos. Quitan 2 globos y luego ponen 12 globos más. Sacan 1 globo. Quitan la mitad. Al final, ¿cuántos globos hay en la caja? → **6**
- **train · nivel 4 · tipo B (desordenado)**: Ana tiene 13 naranjas menos que Ciro. Ciro tiene 17 naranjas. Álvaro tiene 18 más que Clara. Ana tiene el cuádruple que Sofía. Clara tiene 15 naranjas más que Sofía. ¿Cuál es el número de naranjas de Álvaro? → **34**
- **train · nivel 1 · tipo A**: Al empezar el día, en una bolsa hay 10 piedras. Alguien añade 3 piedras a la bolsa. Al final, ¿cuántas piedras hay en la bolsa? → **13**
- **train · nivel 2 · tipo B (desordenado)**: Ana tiene la mitad que Nuria. Tania tiene 20 peonzas menos que Nuria. Tania tiene exactamente 4 peonzas. ¿Cuántas peonzas tiene Ana? → **12**
- **train · nivel 2 · tipo A**: Rocío siempre lleva la cuenta de sus sellos. Rocío guarda 18 sellos en su habitación. Reparte sus sellos en 3 montones iguales y se queda con uno. Regala la mitad de sus sellos. ¿Cuántos sellos tiene Rocío al final? → **3**
- **train · nivel 3 · tipo A**: Una lata contiene 10 piedras. La lata está encima de la mesa. El número de piedras de la lata se triplica. Entonces alguien se lleva 20 piedras de la lata. Al final alguien añade 16 piedras a la lata. Al final, ¿cuántas piedras hay en la lata? → **26**

- **val · nivel 4 · tipo B**: Sabemos que Sara tiene 1 piedra. Sara tiene 8 piedras menos que Mario. Marta tiene 10 piedras más que Mario. Marta tiene 1 piedra más que Laura. Iván tiene la tercera parte de las piedras de Laura. ¿Cuántas piedras tiene Iván? → **6**
- **val · nivel 4 · tipo A**: Jorge tiene 14 piedras. Regala 7 piedras. Más tarde duplica sus piedras. Gana 13 piedras más. Jorge divide sus piedras en 3 grupos iguales y guarda solo uno. ¿Cuántas piedras tiene Jorge al final? → **9**
- **val · nivel 3 · tipo B**: Javier tiene 16 piedras. Raúl tiene la mitad de piedras que Javier. Julia tiene 20 más que Raúl. Lucía tiene 17 piedras menos que Julia. ¿Cuántas piedras tiene Lucía? → **11**
- **val · nivel 2 · tipo B**: Jorge tiene exactamente 6 sellos. Alba tiene la mitad de sellos que Jorge. Alba tiene la tercera parte de los sellos de Ciro. Entonces, ¿cuántos sellos tiene Ciro? → **9**
- **val · nivel 1 · tipo B (desordenado)**: Unos amigos cuentan las flores que tiene cada uno. Mario tiene la mitad de flores que Inés. Mario tiene exactamente 7 flores. ¿Cuál es el número de flores de Inés? → **14**
- **val · nivel 2 · tipo A**: Al empezar el día, en una cesta hay 5 lápices. Alguien añade 15 lápices a la cesta y luego el número de lápices de la cesta se duplica. ¿Cuántos lápices quedan en la cesta? → **40**
- **val · nivel 3 · tipo A**: Al empezar el día, en una bolsa hay 18 naranjas. Alguien añade 11 naranjas a la bolsa. Entonces el número de naranjas de la bolsa se duplica. Alguien se lleva 3 naranjas de la bolsa. Al final, ¿cuántas naranjas hay en la bolsa? → **55**
- **val · nivel 1 · tipo A**: Pablo siempre lleva la cuenta de sus lápices. Pablo guarda 12 lápices en su habitación. Se queda solo con la mitad de sus lápices. ¿Cuántos lápices tiene Pablo ahora? → **6**

- **test_id · nivel 1 · tipo A**: Al principio, Lola tiene 20 cromos. Multiplica por tres sus cromos. ¿Cuántos cromos tiene Lola ahora? → **60**
- **test_id · nivel 1 · tipo B**: Unos amigos cuentan las cartas que tiene cada uno. Sabemos que Julia tiene 15 cartas. Nuria tiene tres veces lo que tiene Julia. ¿Cuál es el número de cartas de Nuria? → **45**
- **test_id · nivel 3 · tipo A**: Nico empieza el día con 17 globos. Regala 15 globos. Nico pierde 1 globo. Entonces duplica sus globos. ¿Con cuántos globos se queda Nico? → **2**
- **test_id · nivel 4 · tipo B (desordenado)**: Víctor tiene 8 sellos menos que Celia. Irene tiene el doble que Víctor. Celia tiene 11 más que Fran. Laura tiene 12 sellos. Fran tiene 16 sellos más que Laura. Entonces, ¿cuántos sellos tiene Irene? → **62**
- **test_id · nivel 3 · tipo B**: Dani tiene exactamente 14 rotuladores. Dani tiene la tercera parte de los rotuladores de Sofía. Sofía tiene 2 rotuladores menos que Vera. Julia tiene 8 rotuladores más que Vera. ¿Cuántos rotuladores tiene Julia? → **52**
- **test_id · nivel 2 · tipo B (desordenado)**: Julia tiene la mitad que Laura. Laura tiene el triple de postales que Pablo. Pablo tiene 18 postales. Entonces, ¿cuántas postales tiene Julia? → **27**
- **test_id · nivel 4 · tipo A**: Óscar empieza el día con 17 peonzas. Gana 8 peonzas más. Compra 17 peonzas. Se queda solo con la mitad de sus peonzas. Gana 5 peonzas más. ¿Cuántas peonzas tiene Óscar al final? → **26**
- **test_id · nivel 2 · tipo A**: Al principio, Teo tiene 5 lápices. Teo multiplica por tres sus lápices y más tarde un vecino le da tantos lápices como ya tenía. ¿Cuántos lápices tiene Teo ahora? → **30**

- **test_tpl · nivel 4 · tipo B (desordenado)**: Raúl tiene 5 bombones más que Luis. Clara tiene 13 bombones más que Javier. Luis tiene 1 bombón. Javier tiene el doble de bombones que Gala. Gala tiene 4 menos que Raúl. Entonces, ¿cuántos bombones tiene Clara? → **17**
- **test_tpl · nivel 2 · tipo A**: Al empezar el día, en una caja hay 14 naranjas. Un niño saca 1 naranja. Más tarde el número de naranjas de la caja se duplica. Al final, ¿cuántas naranjas hay en la caja? → **26**
- **test_tpl · nivel 2 · tipo B (desordenado)**: Irene tiene tantos cuadernos como Ciro y 9 más. Sofía tiene 14 cuadernos. Ciro tiene la mitad que Sofía. ¿Cuál es el número de cuadernos de Irene? → **16**
- **test_tpl · nivel 1 · tipo B**: Sabemos que Tania tiene 20 galletas. Tania tiene el doble de galletas que Irene. ¿Cuántas galletas tiene Irene? → **10**
- **test_tpl · nivel 3 · tipo A**: Una lata contiene 15 cromos. Nadie sabe quién ha dejado allí la lata. Alguien se lleva 6 cromos de la lata. Entonces dejan solo la tercera parte de los cromos. Una niña mete 5 cromos más. ¿Cuántos cromos quedan en la lata? → **8**
- **test_tpl · nivel 4 · tipo A**: En una caja hay 10 rotuladores. Duplican los rotuladores y después dejan solo la cuarta parte de los rotuladores. Quitan 1 rotulador. Una niña mete 13 rotuladores más. ¿Cuántos rotuladores quedan en la caja? → **17**
- **test_tpl · nivel 1 · tipo A**: Una mochila contiene 6 manzanas. Nadie sabe quién ha dejado allí la mochila. Dejan solo la tercera parte de las manzanas. ¿Cuántas manzanas hay ahora en la mochila? → **2**
- **test_tpl · nivel 3 · tipo B**: Julia tiene 9 manzanas. Dora tiene el doble de manzanas que Julia. Vera tiene tantas manzanas como Dora y 7 más. Carlos tiene 3 manzanas más que Vera. ¿Cuántas manzanas tiene Carlos? → **28**

- **test_ood · nivel 5 · tipo B (desordenado)**: Javier tiene el doble que Laura. Raúl tiene 9 flores menos que Hugo. Tania tiene 19 flores. Laura tiene 9 flores menos que Noelia. Noelia tiene 15 más que Tania. Hugo tiene 15 más que Javier. ¿Cuál es el número de flores de Raúl? → **56**
- **test_ood · nivel 6 · tipo A**: Eva empieza el día con 1 postal. Duplica sus postales y más tarde gana 3 postales más. Triplica sus postales. Luego regala 9 postales. Eva pierde 4 postales. Regala 1 postal. ¿Con cuántas postales se queda Eva? → **1**
- **test_ood · nivel 5 · tipo A**: En una cesta hay 16 manzanas. Meten 20 manzanas más. Duplican las manzanas. Quitan 8 manzanas y luego meten 16 manzanas más. Quitan 3 manzanas. Al final, ¿cuántas manzanas hay en la cesta? → **77**
- **test_ood · nivel 6 · tipo B (desordenado)**: Sara tiene 1 globo. Sofía tiene 14 globos más que Sara. Sofía tiene el triple que Irene. Andrés tiene el doble que Álvaro. Luis tiene 5 globos menos que Lucía. Lucía tiene el doble de globos que Irene. Álvaro tiene 6 globos más que Luis. ¿Cuántos globos tiene Andrés? → **22**

