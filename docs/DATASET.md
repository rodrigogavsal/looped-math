# Dataset de problemas matemáticos en castellano

**Proyecto:** Práctica 1 de PLN I (Universidad de Alicante): Looped Transformer sobre el GPT de Raschka.
**Parte:** 1 · Generación y verificación del dataset.
**Código:** `src/generate_data.py` (generador), `src/tokenizer.py` (tokenizador y formato), `src/text_solver.py` (verificación independiente), `src/calibrate.py` (calibración con un GPT), `src/eval_templates.py` (exactitud por redacción con un checkpoint), `tests/test_data.py` (tests).

> **Cómo leer este documento.** Está pensado para dos públicos: los miembros del equipo y un asistente de IA que vaya a ayudar con otra parte de la práctica (modelos, entrenamiento, análisis). Las secciones 1–3 dan el contexto y las decisiones de diseño; la 4–7 describen el dataset y su formato con precisión de contrato; la 8 explica cómo generarlo y calibrarlo; la 9 recoge las hipótesis que queremos poner a prueba; la 10 tiene notas para cada parte del equipo; la 11 resume las limitaciones. Si eres una IA ayudando con otra parte: respeta el formato de las secciones 5–6 tal cual, porque el resto del equipo depende de él.

---

## 1. Contexto

Partimos del GPT del libro *Build a Large Language Model (From Scratch)* (Raschka, 2024): arquitectura GPT‑2 (pre‑LayerNorm, GELU, red feed‑forward de 4·d, embeddings posicionales **absolutos y aprendidos**), reducida a `emb_dim = 256`, 8 cabezas y contexto de 512 tokens. La ampliación es un **looped transformer**: en lugar de apilar L bloques con pesos distintos, se aplica **un único bloque N veces con los mismos pesos** (`for t in range(N): h = block(h)`). Entrenamos desde cero GPTs normales de 1, 2, 4, 8 y 12 capas y looped de 2, 4, 8 y 12 iteraciones, con tres semillas cada uno, y comparamos rendimiento y comportamiento interno.

La tarea son **problemas de matemáticas en castellano con respuesta numérica** y dificultad controlada. El modelo debe dar el número directamente, **sin escribir razonamiento intermedio**: todo el cálculo ocurre "por dentro" de la red. Eso convierte la profundidad (capas o iteraciones del bucle) en el recurso que limita lo que el modelo puede resolver, que es exactamente lo que queremos estudiar.

El trabajo se reparte en cuatro partes que avanzan en paralelo contra un **contrato común** (formato de datos, interfaz del modelo, formato de resultados):

| Parte | Qué hace |
|---|---|
| 1 · Datos | Este dataset: generador, particiones, verificación y calibración. |
| 2 · Modelos y entorno | Repositorio, `model.py` (GPT y Looped con salida de estados y atenciones), tests, configuraciones, benchmark de memoria. |
| 3 · Entrenamiento y evaluación | DataLoader, bucle de entrenamiento, evaluación por generación, lanzador de experimentos. |
| 4 · Análisis y artículo | Métricas de representaciones y atención, logit lens, figuras, tablas, artículo ACL de 4 páginas. |

---

## 2. Por qué el dataset es así

**Sintético y no real.** Un modelo de unos pocos millones de parámetros entrenado desde cero no aprende a resolver problemas en lenguaje libre como GSM8K: sacaría 0 % en todo y no habría nada que comparar. Un generador con plantillas da respuestas exactas, dificultad controlada, tantos ejemplos como haga falta y una traza de la solución para analizar errores.

**La dificultad debe ser profundidad de verdad.** Definimos *nivel = número de operaciones encadenadas*. Pero si un problema solo tiene sumas y restas sobre una cantidad ("tiene 12, gana 7, pierde 3"), el orden da igual: el modelo puede sumar todos los números con su signo de una sola pasada, y el nivel 6 no exigiría más profundidad que el 1. Por eso:

- Hay operaciones que **no conmutan** con la suma: el doble, el triple, la mitad, la tercera o cuarta parte. (12 − 5) × 2 ≠ 12 − 5 × 2, así que hay que calcular en orden.
- **A partir del nivel 2, todo problema tiene al menos una operación multiplicativa.**
- **Nunca hay dos operaciones seguidas que se anulen** (el doble y luego la mitad, sumar 5 y luego restar 5). Esos pares dejan el valor como estaba: el problema tendría menos pasos reales que su nivel y la respuesta saldría copiando un número del texto. Antes de prohibirlos, en el nivel 2 el 10,4 % de las respuestas era el número inicial y la heurística "contesta el primer número" acertaba el 10,2 %; ahora acierta el 3,8 % con el rango actual (1,9 % con el rango original de la sección 3).
- En el tipo B, cada cantidad se define **respecto a otra persona**, de modo que hay que resolver una cadena de referencias (multi‑hop). En la mitad de los problemas B las frases están **desordenadas**, así que tampoco basta con leer de izquierda a derecha.

---

## 3. Decisión de diseño: un rango cerrado de valores

**Qué decidimos.** Todos los operandos de sumas y restas (y el valor inicial) están entre 1 y 20, y **todas las cantidades del problema (iniciales, intermedias y finales) entre 1 y 99**. Las divisiones son siempre exactas. Las respuestas son, por tanto, enteros de 1 a 99 (como mucho dos cifras). Están en `MIN_VAL`, `MAX_VAL` y `MAX_OPERAND` al principio de `src/generate_data.py`.

**Por qué.**

- **Separar razonamiento de aritmética.** Con un tokenizador por caracteres, sumar números grandes con llevadas ya es un cálculo de varios pasos para el modelo. Si los números fueran grandes, un fallo podría deberse a la aritmética y no a encadenar operaciones, y no sabríamos distinguirlo. Con números pequeños, la variable que controlamos es el número de pasos.
- **Experimento controlado.** Todos los niveles, incluido el OOD (5–6), comparten el mismo rango. Así, cuando un modelo falla más en el nivel 6 que en el 2, sabemos que es por los pasos y no porque los números sean distintos.
- **Texto natural.** Evitamos cantidades negativas o cero ("tiene 0 canicas"), que sonarían raras en castellano.
- **Calibración (sección 8.1).** Empezamos con operandos de 1 a 50 y valores de 1 a 200. Con ese rango, un GPT de 4 capas entrenado 20 000 pasos solo llegaba al 64,7 % en el nivel 1 y se quedaba en el azar en los niveles 3–4 (6,9 % y 5,8 %), así que no habría permitido distinguir ninguna arquitectura. Con 1–20 y 1–99, el mismo modelo y los mismos pasos dan 94,7 / 47,3 / 18,6 / 14,0 %, y con 42 000 pasos 97 / 80 / 51 / 37 %. **El rango se redujo por la calibración, no por estética.**

**Consecuencias que hay que tener presentes** (y declarar en el artículo):

1. **El modelo aprende el rango.** Nunca ve una respuesta de 0 ni mayor que 99. No afecta a nuestras comparaciones (todo está dentro del rango), pero **no podemos afirmar que el modelo "sabe aritmética"** en general, solo que resuelve problemas dentro de este rango. Con números mayores fallaría por la magnitud, no por la profundidad.
2. **Las restricciones filtran un poco de información.** "Triplica" solo aparece si el valor era ≤ 33; "la mitad", si era par; "la tercera parte", si era múltiplo de 3. No revela la respuesta, pero existe.
3. **Distribución sesgada hacia números pequeños.** El 39 % de las respuestas de `train` tiene una cifra y el 1 es la respuesta más frecuente (7,4 %). Por eso hay que reportar siempre las referencias de la sección 7.3: un 7 % de exactitud no es aprendizaje.
4. **Poca variedad de aritmética en los niveles bajos.** En el nivel 1 solo hay unas 650 combinaciones distintas de números y operación en `train` (sección 7.2).

---

## 4. Qué contiene

### 4.1 Tipos de problema (50 % cada uno en todos los niveles)

**Tipo A · Secuencia.** Una cantidad cambia paso a paso. Dos marcos al 50 %:

- *Persona*: alguien tiene objetos y compra, regala, pierde, duplica, reparte…
- *Recipiente*: una caja, cesta, bolsa, mochila, lata o bandeja a la que meten, sacan, duplican…

**Tipo B · Cadena de comparaciones.** La primera persona tiene una cantidad; cada persona siguiente se define respecto a la anterior; se pregunta por la última. Incluye formulaciones **inversas** ("Ana tiene 3 menos que Luis" define a Luis a partir de Ana), que obligan a invertir la relación. El 50 % de los problemas B tiene las frases desordenadas (`"shuffled": true`).

### 4.2 Niveles

| Nivel | Operaciones | Dónde aparece |
|---|---|---|
| 1–4 | 1 a 4 | `train`, `val`, `test_id`, `test_tpl` |
| 5–6 | 5 y 6 | Solo `test_ood` (nunca vistos al entrenar) |

### 4.3 Ejemplos reales del generador

| Nivel | Tipo | Problema | Respuesta |
|---|---|---|---|
| 1 | A | Al empezar el día, en una bolsa hay 1 tornillo. El número de tornillos de la bolsa se triplica. ¿Cuántos tornillos hay ahora en la bolsa? | 3 |
| 1 | B desord. | Unos amigos cuentan los libros que tiene cada uno. Edu tiene la mitad de libros que Elena. Sabemos que Edu tiene 19 libros. Entonces, ¿cuántos libros tiene Elena? | 38 |
| 2 | A | Fran tiene 2 rotuladores. Gana 13 rotuladores más. Fran duplica sus rotuladores. ¿Cuántos rotuladores tiene Fran al final? | 30 |
| 2 | B desord. | Mario tiene la mitad de bombones que Pedro. Pedro tiene 2 bombones. Dora tiene 2 bombones más que Mario. ¿Cuántos bombones tiene Dora? | 3 |
| 3 | A | Una caja contiene 11 naranjas. Alguien añade 14 naranjas a la caja y luego el número de naranjas de la caja se duplica. Alguien se lleva 14 naranjas de la caja. Al final, ¿cuántas naranjas hay en la caja? | 36 |
| 3 | B desord. | Ana tiene 6 más que Óscar. Víctor tiene 20 botones. Víctor tiene el cuádruple que Óscar. Dora tiene 3 botones más que Ana. Entonces, ¿cuántos botones tiene Dora? | 14 |
| 4 | A | Óscar tiene 1 lápiz. Gana 18 lápices más. Regala 4 lápices y gana 17 lápices más. Al final regala la mitad de sus lápices. ¿Cuántos lápices tiene Óscar ahora? | 16 |
| 4 | B | Marta tiene 6 manzanas. Nuria tiene 1 manzana más que Marta. Fran tiene 2 manzanas menos que Nuria. Teo tiene el doble que Fran. Tania tiene 15 más que Teo. ¿Cuántas manzanas tiene Tania? | 25 |
| 5 | A | Jorge guarda 15 figuras en su habitación. Le da 11 figuras a su tío y más tarde encuentra 5 figuras más en un cajón. Su tío le regala 10 figuras. Entonces encuentra 5 figuras más en un cajón. Reparte sus figuras en 3 montones iguales y se queda con uno. ¿Cuántas figuras tiene Jorge al final? | 8 |
| 6 | B desord. | Nico tiene 4 más que Nuria. Javier tiene el doble que Nico. Sara tiene 7 globos. Gala tiene el triple de globos que Javier. Rocío tiene la mitad de globos que Gala. Nuria tiene 4 globos más que Sara. Vera tiene 7 globos menos que Rocío. ¿Cuál es el número de globos de Vera? | 38 |

### 4.4 Operaciones

| Operación | Símbolo en la traza | Condición | Ejemplos de redacción (A / B) |
|---|---|---|---|
| Sumar k | `"+", k` | 1 ≤ k ≤ 20, resultado ≤ 99 | "compra 7 canicas" / "Luis tiene 7 más que Ana" |
| Restar k | `"-", k` | 1 ≤ k ≤ 20, resultado ≥ 1 | "regala 7 canicas" / "A Luis le faltan 7 para tener tantas como Ana" |
| Doble | `"*", 2` | resultado ≤ 99 | "duplica sus canicas" / "Luis tiene el doble que Ana" |
| Triple | `"*", 3` | resultado ≤ 99 | "multiplica por tres sus canicas" / "Luis tiene tres veces lo que tiene Ana" |
| Mitad | `"/", 2` | división exacta | "regala la mitad de sus canicas" / "Ana tiene el doble que Luis" (inversa) |
| Tercera / cuarta parte | `"/", 3` o `"/", 4` | división exacta | "reparte sus canicas en 3 montones iguales y se queda con uno" / "Ana tiene el cuádruple que Luis" (inversa) |

Frecuencia en `train`: suma 27,9 %, resta 25,2 %, doble 17,8 %, triple 11,5 %, mitad 9,7 %, tercera parte 4,4 %, cuarta parte 3,5 %.

### 4.5 Variedad del castellano

- 48 nombres (con tildes: Óscar, Inés, Álvaro…), 24 objetos con su género y su plural (canica/canicas, lápiz/lápices, bombón/bombones…), 6 recipientes, 14 personajes secundarios ("su abuela", "un vecino"…).
- Redacciones por operación: 23 en el marco persona, 20 en el marco recipiente y 22 en el tipo B; además de 3–4 inicios y 3 preguntas por marco.
- Conectores ("luego", "después", "más tarde", "entonces", "al final"), sujeto explícito u omitido, y acciones unidas con "y". Cada conector se usa como mucho una vez por problema, y nunca se repite la misma redacción en dos operaciones seguidas si hay alternativa.
- **Concordancia** de género ("¿cuántas canicas?" / "¿cuántos cromos?", "tantas/tantos") y de número ("1 canica", "se le pierde 1 canica", "le falta 1 lápiz"). Comprobada en los 224 000 problemas: 0 errores.
- **Estilo** compacto o *verbose* (redacciones largas y una frase de ambientación sin números). La probabilidad de *verbose* baja con el nivel (80 % en el nivel 1, 9 % en el 6) para que la longitud del texto no delate el nivel.
- No hay distractores (datos numéricos que no se usan). Las frases de ambientación no contienen números.

### 4.6 Redacciones reservadas (test de plantillas no vistas)

Por cada operación y marco se reserva **una redacción** (18 en total, listadas en `data/data_card.md` y en `HELD_IDS`). Esas redacciones **nunca aparecen** en `train`, `val`, `test_id` ni `test_ood`, y **cada problema de `test_tpl` usa al menos una**. Así podemos medir si el modelo entiende el lenguaje o solo ha memorizado frases concretas.

---

## 5. Ficheros y formato (contrato)

### 5.1 Ficheros

| Fichero | Niveles | Problemas | Uso |
|---|---|---|---|
| `data/train.jsonl` | 1–4 | 200 000 (50 000 por nivel) | Entrenamiento |
| `data/val.jsonl` | 1–4 | 4 000 (1 000 por nivel) | Seguir el entrenamiento, elegir checkpoint, calibrar |
| `data/test_id.jsonl` | 1–4 | 8 000 (2 000 por nivel) | Resultados principales |
| `data/test_tpl.jsonl` | 1–4 | 8 000 (2 000 por nivel) | Generalización a redacciones no vistas |
| `data/test_ood.jsonl` | 5–6 | 4 000 (2 000 por nivel) | Generalización a más pasos |
| `data/data_card.md` | — | — | Estadísticas, referencias y ejemplos |
| `data/sample/` | — | 50 por nivel y fichero | Muestra pequeña para desarrollar |

`data/` no se sube a git (salvo `data/sample/`): se regenera en unos 25 segundos con el comando de la sección 8. El `train.jsonl` ocupa unos 80 MB. Los ficheros se escriben siempre con finales de línea `\n`, así que salen **idénticos byte a byte en Windows y en Linux**: el equipo puede comparar huellas (`sha256sum` / `Get-FileHash`) para confirmar que todos usan el mismo dataset. Huellas SHA-256 de los ficheros generados con los valores por defecto (semilla 1234):

```
29fa6e5b66f6c51425481337bc2524a5149f0026f5009933825a7b4ff98f0df7  train.jsonl
1e63c5f985f117f422acd4e2af2c7eec59b2a190791406f48f2ed148b9065906  val.jsonl
6f34c0ca38f23a0713caf6bb47ef6c5453cbdb32a3eed5981af52cac03697da0  test_id.jsonl
fa0f833aa75c8fa9c81bdfb27ad74be87a13e908dca451b9c9b8847a07a15062  test_tpl.jsonl
af29c618af284eba9db8ea2b30c70d293bd7c8c0906bf0214669974e68fcaad7  test_ood.jsonl
```

### 5.2 Una línea de JSONL

```json
{"id": "val-L2-000367", "level": 2, "type": "B", "shuffled": true, "style": "verbose",
 "question": "Dora tiene la mitad de conchas que Lucía. Iván tiene tres veces lo que tiene Lucía. Sabemos que Dora tiene 14 conchas. Entonces, ¿cuántas conchas tiene Iván?",
 "answer": 84, "trace": [14, "*", 2, "*", 3], "template": "B|bs_sabemos|bm2_inv|bm3_veces|bq_entonces"}
```

| Campo | Tipo | Significado |
|---|---|---|
| `id` | str | `<fichero>-L<nivel>-<nº>`, único en todo el dataset |
| `level` | int | Número de operaciones (1–6) |
| `type` | `"A"` / `"B"` | Secuencia / cadena de comparaciones |
| `shuffled` | bool | Frases desordenadas (solo puede ser `true` en el tipo B) |
| `style` | `"compact"` / `"verbose"` | Estilo de redacción |
| `question` | str | El enunciado, que termina en "?" |
| `answer` | int | Respuesta, entre 1 y 99 |
| `trace` | list | `[valor_inicial, op, arg, op, arg, …]` evaluada de izquierda a derecha. En el tipo B sigue el orden de la cadena, no el orden de las frases. Símbolos: `"+"`, `"-"`, `"*"`, `"/"` |
| `template` | str | Marco e ids de las redacciones usadas (`A-persona`, `A-recipiente` o `B`), separados por `|` |

### 5.3 Diferencias respecto al contrato común original

Son añadidos y no rompen nada: el fichero `test_tpl.jsonl` y los campos `type`, `shuffled` y `style`. Además, **ningún fichero de evaluación tiene ejemplos más largos que el más largo de `train`** (411 tokens).

---

## 6. Cómo lo ve el modelo

### 6.1 Texto

```
"Pregunta: " + question + "\nRespuesta: " + str(answer)      y después el token <eos>
```

- **Entrenamiento:** la pérdida se calcula **solo** sobre los caracteres de la respuesta y el `<eos>`. Todas las posiciones anteriores, y el relleno, llevan objetivo `-100`.
- **Evaluación:** el modelo recibe `format_prompt(question)` (todo hasta `"Respuesta: "` incluido) y genera de forma greedy hasta `<eos>` o un máximo de 5 caracteres. Se extrae el entero; si no hay entero, cuenta como fallo.
- **Atajo equivalente para medir la exactitud:** con teacher forcing, un problema es correcto si el argmax acierta *todos* los caracteres de la respuesta y el `<eos>`. Da exactamente el mismo resultado que generar (si acierta cada carácter dado el prefijo correcto, el greedy produce justo ese prefijo) y es mucho más rápido. Lo usa `src/calibrate.py`.
- Funciones auxiliares en `src/tokenizer.py`: `format_prompt`, `format_example`, `example_length`.

### 6.2 Tokenizador (por caracteres, vocabulario fijo)

- `<pad>` = 0, `<eos>` = 1, `<unk>` = 2, y después 94 caracteres: espacio, salto de línea, dígitos, a–z, A–Z, `áéíóúüñÁÉÍÓÚÜÑ` y `.,;:¿?¡!()-"'+×=`. **`vocab_size` = 97.**
- Cada dígito es un token: los números se leen cifra a cifra, sin las irregularidades del BPE de GPT‑2.
- `MAX_LEN = 512` tokens por ejemplo, incluido el `<eos>` (= `context_length` del modelo).

### 6.3 Longitudes (tokens, incluido `<eos>`)

| Nivel | Media | Máximo |
|---|---|---|
| 1 | 154 | 241 |
| 2 | 190 | 301 |
| 3 | 219 | 355 |
| 4 | 244 | 411 |
| 5 (OOD) | 263 | 408 |
| 6 (OOD) | 289 | 411 |

Las longitudes se solapan mucho entre niveles, pero la media sigue creciendo con el nivel (inevitable: más operaciones son más frases).

---

## 7. Garantías y referencias

### 7.1 Garantías (comprobadas por `tests/test_data.py`, 20 tests)

1. Cada `answer` es el resultado de evaluar su `trace`, y todas las cantidades intermedias están entre 1 y 99; las divisiones son exactas.
2. **Resolviendo cada problema solo a partir del texto** (`src/text_solver.py`, que no usa ni la traza ni el generador) se obtiene la misma respuesta, y el número de operaciones que se leen en el texto es igual a `level`. Comprobado en los 224 000 problemas: 0 discrepancias.
3. `level` es igual al número de operaciones de la traza.
4. A partir del nivel 2 hay al menos una operación multiplicativa.
5. Nunca hay dos operaciones seguidas que se anulen.
6. Todos los operandos de sumas y restas, y el valor inicial, aparecen como número en el texto.
7. Todos los caracteres están en el vocabulario; ningún ejemplo supera 512 tokens.
8. Ningún ejemplo de evaluación es más largo que el ejemplo más largo de `train`.
9. Ninguna pregunta se repite, ni dentro de un fichero ni entre ficheros. Los `id` son únicos.
10. Las redacciones reservadas solo aparecen en `test_tpl`, y todo problema de `test_tpl` usa al menos una.
11. Exactamente 50 % de tipo A y 50 % de tipo B en cada nivel de cada fichero; ~50 % de los B desordenados.
12. La generación es determinista (la misma semilla produce ficheros idénticos) y usa finales de línea `\n`.
13. El tokenizador codifica y decodifica sin pérdida.

Además se ha revisado a mano una muestra de cada tipo, nivel y fichero (aritmética y castellano).

### 7.2 Solapamiento de trazas con train

Porcentaje de problemas de evaluación cuya traza (mismos números y operaciones) aparece también en `train`, con otro texto:

| Fichero | Nivel 1 | Nivel 2 | Nivel 3 | Nivel 4 | Nivel 5 | Nivel 6 |
|---|---|---|---|---|---|---|
| test_id | 100 % | 100 % | 48 % | 10 % | — | — |
| test_tpl | 100 % | 100 % | 49 % | 11 % | — | — |
| test_ood | — | — | — | — | 0 % | 0 % |

No es fuga de datos: el texto es distinto y el modelo tiene que entenderlo. Pero en los niveles 1–2 la aritmética concreta ya se ha visto (en el nivel 1 hay unas 650 combinaciones distintas en `train`; en el nivel 2, prácticamente todas las de evaluación están ya en `train`), así que esos niveles miden sobre todo **entender el enunciado**; desde el nivel 3, y sobre todo en el nivel 4 y el OOD, el modelo tiene que **calcular algo que no ha visto**. Conviene tenerlo en cuenta al interpretar las curvas por nivel.

### 7.3 Referencias: heurísticas que no razonan

Exactitud que se obtendría sin resolver nada. Cualquier modelo tiene que superarlas con claridad, y las tablas del artículo deberían incluirlas.

| Fichero | Nivel | Siempre la respuesta más frecuente (1) | Primer número del texto | Último número del texto |
|---|---|---|---|---|
| test_id | 1 | 7,6 % | 0,6 % | 3,2 % |
| test_id | 2 | 7,5 % | 3,8 % | 5,2 % |
| test_id | 3 | 6,2 % | 3,9 % | 5,5 % |
| test_id | 4 | 5,4 % | 3,0 % | 4,8 % |
| test_ood | 5 | 4,7 % | 3,2 % | 3,8 % |
| test_ood | 6 | 5,5 % | 2,6 % | 4,3 % |

---

## 8. Cómo generarlo, verificarlo y calibrarlo

Desde la raíz del repositorio, con Python ≥ 3.8. El generador no usa librerías externas; los tests necesitan `pytest` y la calibración, `torch`.

```bash
python -m src.generate_data --out data                    # dataset completo (~25 s)
python -m src.generate_data --out data/sample --sample    # muestra de 50 por nivel y fichero
python -m pytest tests/test_data.py -q                    # tests sobre un dataset pequeño generado al vuelo
DATA_DIR=data python -m pytest tests/test_data.py -q      # tests sobre el dataset real (Linux / Git Bash)
python -m src.text_solver data/*.jsonl                    # resolvedor por texto (también lo ejecutan los tests)
python -m src.calibrate --steps 42000 --eval_every 3000   # calibración de referencia (GPT de 4 capas, ~77 min en una RTX 3050 Ti)
python -m src.eval_templates --load results/calibration_L4_d256_s0.pt   # exactitud por redacción reservada
```

En PowerShell, el test sobre el dataset real es `$env:DATA_DIR="data"; python -m pytest tests/test_data.py -q`, y el resolvedor, `python -m src.text_solver (Get-ChildItem data\*.jsonl)`.

Opciones del generador: `--seed` (por defecto 1234), `--n_train`, `--n_val`, `--n_test` (tamaños **por nivel**).

### 8.1 Calibración

Antes de lanzar la matriz de experimentos hay que comprobar que la dificultad deja margen para que los modelos se distingan. `python -m src.calibrate` entrena un GPT de 4 capas (`emb_dim` 256, 8 cabezas, **3 337 216 parámetros**, tokenizador por caracteres, AdamW, lr 1e‑3 con warmup de 200 pasos y coseno hasta el 10 %, batch 64) y mide la exactitud en `val` por nivel y por tipo cada `--eval_every` pasos. Al terminar evalúa `test_id`, `test_tpl` y `test_ood` y guarda el resultado (`results/<out>_L4_d256_s0.json`) y el checkpoint (`.pt`). Con `--load` se reevalúa un checkpoint sin entrenar.

- **Objetivo aproximado:** ~95 % en el nivel 1 y 20–40 % en el nivel 4.
- **Demasiado fácil** (nivel 4 por encima del ~60 %): subir `MAX_OPERAND`, dar más peso a las operaciones multiplicativas en `OP_WEIGHTS`, o ambas.
- **Demasiado difícil** (nivel 1 por debajo del ~60 %): primero comprobar que la curva no sigue subiendo (si sube, alargar `--steps`); si está estancada, bajar `MAX_OPERAND` o `MAX_VAL`.
- Los parámetros están al principio de `src/generate_data.py`. Tras cambiarlos: regenerar, pasar los tests y volver a calibrar.

**Resultado de la calibración (configuración definitiva: operandos 1–20, valores 1–99).** Una ejecución, semilla 0, 42 000 pasos (≈13 épocas de `train`; 77,5 min en una RTX 3050 Ti de 4 GB con bf16). Exactitud exacta de la respuesta:

| Fichero | Nivel 1 | Nivel 2 | Nivel 3 | Nivel 4 | Nivel 5 | Nivel 6 | Total |
|---|---|---|---|---|---|---|---|
| val | 97,4 % | 80,4 % | 48,9 % | 38,0 % | — | — | 66,2 % |
| test_id | 97,0 % | 80,8 % | 50,5 % | 35,9 % | — | — | 66,0 % |
| test_tpl | 41,9 % | 34,3 % | 20,1 % | 13,6 % | — | — | 27,5 % |
| test_ood | — | — | — | — | 23,5 % | 16,7 % | 20,1 % |

Por tipo (A / B desordenado / B ordenado): `test_id` 79,2 / 42,0 / 64,4 %; `test_tpl` 37,1 / 13,2 / 22,2 %; `test_ood` 28,4 / 6,6 / 17,4 %. Las referencias que no razonan están en la sección 7.3 (4,7–7,6 %).

**Efecto de la profundidad (misma configuración, 1 capa).** Un GPT de **1 capa** (970 240 parámetros, 23,4 min) con los mismos 42 000 pasos y la semilla 0:

| Fichero | Nivel 1 | Nivel 2 | Nivel 3 | Nivel 4 | Nivel 5 | Nivel 6 | Total |
|---|---|---|---|---|---|---|---|
| val | 83,6 % | 33,3 % | 9,1 % | 8,0 % | — | — | 33,5 % |
| test_id | 85,5 % | 34,0 % | 9,2 % | 7,4 % | — | — | 34,0 % |
| test_tpl | 35,1 % | 18,3 % | 6,9 % | 4,3 % | — | — | 16,2 % |
| test_ood | — | — | — | — | 5,6 % | 4,4 % | 5,0 % |

Con 1 capa los niveles 3–4 y el OOD quedan cerca del azar (referencias de la sección 7.3: 4,7–6,2 % en los niveles 3–6; el nivel 3–4 de 1 capa, con 8–9 %, está solo un poco por encima), el nivel 2 se queda en un tercio y el nivel 1 no pasa del 86 %. Con 4 capas: 97 / 81 / 51 / 36 % en `test_id`. **La diferencia es muy superior al ruido de ±2 puntos (punto 4 de abajo): el dataset exige profundidad desde el nivel 2.** Matiz: 1 capa tiene también ~3,4 veces menos parámetros que 4 capas, así que esta comparación mezcla profundidad y tamaño; separarlos es el objetivo de H2 y H3.

Historial de la decisión del rango (misma configuración de modelo, semilla 0, solo `val`):

| Rango | Pasos | Nivel 1 | Nivel 2 | Nivel 3 | Nivel 4 |
|---|---|---|---|---|---|
| operandos 1–50, valores 1–200 | 20 000 | 64,7 % | 18,4 % | 6,9 % | 5,8 % |
| operandos 1–20, valores 1–99 | 20 000 | 94,7 % | 47,3 % | 18,6 % | 14,0 % |
| operandos 1–20, valores 1–99 | 42 000 | 97,4 % | 80,4 % | 48,9 % | 38,0 % |

**Cómo leerlo:**

1. **`val` y `test_id` coinciden** (66,2 % y 66,0 %): no hay sobreajuste y `val` sirve para decidir sin tocar los test.
2. **La dificultad crece con el nivel y deja margen** en los niveles 3–4 y en el tipo B desordenado (43 %, frente al 78 % del tipo A). Los niveles 1–2 están casi en el techo, así que discriminarán poco entre modelos grandes.
3. **El despegue es tardío.** En todas las ejecuciones los niveles 2–4 están cerca del azar durante los primeros ~9 000 pasos y luego suben; los niveles 3–4 siguen mejorando hasta ~35 000 pasos y después se aplanan. Con menos pasos, la comparación entre arquitecturas mediría sobre todo velocidad de aprendizaje.
4. **Ruido entre ejecuciones: unos ±2 puntos por nivel.** Dos ejecuciones idénticas (semilla 0) dieron 96,9 / 79,6 / 50,8 / 36,7 % y 97,4 / 80,4 / 48,9 / 38,0 %, y la iteración en la que despega varía (a 6 000 pasos, nivel 1 en 34 % y en 40 %). No interpretar diferencias de menos de 3–4 puntos entre modelos, comparar siempre al final del entrenamiento y usar las 3 semillas.
5. **`test_tpl` cae mucho** (97 % → 42 % en el nivel 1). Un modelo de este tamaño, entrenado desde cero solo con este dataset, no generaliza bien a redacciones nuevas. Es una limitación real de este tipo de modelo, no un defecto del test: los valores absolutos serán bajos para todos y lo que interesa es la diferencia entre arquitecturas. `src/eval_templates.py` desglosa la exactitud por cada redacción reservada y la compara, en el nivel 1, con la de las redacciones vistas de la misma operación.
6. **`test_ood` degrada de forma gradual** (38,0 % en el nivel 4, 23,5 % en el 5, 16,7 % en el 6): cada operación extra multiplica la exactitud por ~0,6–0,8, igual que entre los niveles 1–4. No hay un precipicio, así que `test_ood` no es imposible y deja margen para ver si el looped mejora la extrapolación.

**Coste.** En la RTX 3050 Ti, 42 000 pasos tardan 23,4 min con 1 capa y 77,5 min con 4 (≈ 5 min fijos + 18 min por capa). **Extrapolando** linealmente (no medido): ~41 min con 2 capas, ~150 min con 8 y ~220 min con 12; un looped de N iteraciones cuesta aproximadamente lo mismo que un GPT de N capas. La matriz completa (GPT 1, 2, 4, 8, 12 y looped 2, 4, 8, 12, con 3 semillas) serían del orden de **50 horas en esa GPU**, y con 12 capas puede no caber en 4 GB con batch 64 (habría que bajar el batch o usar checkpointing de activaciones). Hay que fijar el presupuesto de pasos y repartir las ejecuciones entre el equipo al planificar la matriz.

---

## 9. Hipótesis que queremos poner a prueba

Cada hipótesis indica qué se compara, con qué fichero y métrica, y qué esperamos. Los resultados negativos también son resultados: si una hipótesis no se cumple, se cuenta y se analiza.

**Notación.** GPT‑L: GPT normal de L capas distintas. Looped‑N: un bloque aplicado N veces. Todas las comparaciones con el mismo `emb_dim`, los mismos datos y los mismos pasos, con media ± desviación de 3 semillas.

### H1 · Los problemas más largos necesitan más profundidad

- **Qué:** la exactitud cae con el nivel en todos los modelos, y el nivel que un modelo es capaz de resolver crece con su profundidad (capas o iteraciones).
- **Cómo:** exactitud por nivel en `test_id`, frente a la profundidad (1, 2, 4, 8, 12). Para cada nivel, el **mínimo de loops necesario** = el menor N que alcanza el 95 % de la mejor exactitud de ese nivel.
- **Esperado:** el mínimo de loops crece con el nivel. Si no crece, el dataset no está exigiendo profundidad (ver sección 2) y hay que revisarlo.
- **Referencia (GPT, `test_id`, sección 8.1):** con 1 capa 85 / 34 / 9 / 7 % y con 4 capas 97 / 81 / 51 / 36 % en los niveles 1–4. La profundidad ya importa claramente entre 1 y 4; falta ver si 8 y 12 saturan los niveles 1–4 y solo se distinguen en `test_ood` y en el tipo B desordenado.

### H2 · Compartir pesos cuesta poco a igual cómputo

- **Qué:** con la misma profundidad (mismos FLOPs), Looped‑N se queda cerca de GPT‑N aunque tenga N veces menos parámetros en los bloques.
- **Cómo:** Looped‑N frente a GPT‑N en `test_id`, por nivel. Comparación opcional con GPT‑4 de `d = 128`, que tiene los mismos parámetros y la misma profundidad que Looped‑4.
- **Esperado:** diferencia pequeña. Si la diferencia es grande, el bloque compartido no tiene capacidad suficiente para lo que hacen varias capas distintas.

### H3 · Repetir el bloque compensa la falta de parámetros

- **Qué:** con los mismos parámetros, Looped‑N supera claramente a GPT‑1.
- **Cómo:** Looped‑N frente a GPT‑1 en `test_id`, por nivel.
- **Esperado:** ganancia creciente con el nivel. Es el argumento principal a favor del looped: más cómputo sin más parámetros.

### H4 · Generalización a problemas más largos (OOD)

- **Qué:** sobre niveles 5–6, que nunca se vieron al entrenar, el looped generaliza al menos tan bien como el GPT de la misma profundidad.
- **Cómo:** exactitud en `test_ood`, comparada con la del nivel 4 de `test_id`.
- **Esperado:** caída para todos; menor caída en el looped, porque aplica el mismo "paso de razonamiento" repetidamente.
- **Referencia (GPT‑4, sección 8.1):** 36 % en el nivel 4, 23,5 % en el 5 y 16,7 % en el 6.

### H5 · Más loops en test ayudan en problemas más difíciles

- **Qué:** a un looped entrenado con N iteraciones se le pueden dar N' ≠ N en inferencia. Un GPT no puede añadir capas.
- **Cómo:** exactitud con N' = 1…2N, por nivel, en `test_id` y `test_ood` (campo `loops_sweep` de los resultados).
- **Esperado:** con N' < N la exactitud cae, más en los niveles altos. Con N' > N puede mejorar en los niveles 5–6 o degradarse; sin entrenamiento específico para profundidad variable lo segundo es posible, y también es un resultado que hay que reportar.

### H6 · El desorden y las referencias exigen más profundidad

- **Qué:** el tipo B desordenado es más difícil que el B ordenado y que el A, y necesita más loops.
- **Cómo:** exactitud por nivel separando `type` y `shuffled`; mínimo de loops de cada grupo.
- **Esperado:** B desordenado < B ordenado ≈ A. El loop debería ayudar más en el B desordenado.

### H7 · El modelo razona, no memoriza frases

- **Qué:** la exactitud en redacciones no vistas es parecida a la de redacciones vistas.
- **Cómo:** `test_tpl` frente a `test_id`, por nivel.
- **Esperado:** caída pequeña. Si la caída es grande, el modelo ha aprendido plantillas concretas y no la operación.
- **Referencia (GPT‑4, sección 8.1):** la caída es grande (66,0 % → 27,5 %; 97 % → 42 % en el nivel 1), así que probablemente lo será para todos los modelos. Se interpreta la diferencia entre arquitecturas, no el valor absoluto. Desglose por redacción: `src/eval_templates.py`.

### H8 · Qué pasa dentro del bucle (representaciones)

- **Qué:** en el looped, el estado cambia mucho en las primeras iteraciones y poco en las últimas (se acerca a un punto fijo), y la respuesta correcta "aparece" en una iteración que depende del nivel. En el GPT, el patrón por capas es distinto.
- **Cómo:** sobre un conjunto fijo de `test_id` y `test_ood`, en cada iteración o capa, en la posición donde empieza la respuesta:
  - cambio relativo ‖hₜ − hₜ₋₁‖ / ‖hₜ₋₁‖ y coseno entre iteraciones consecutivas;
  - norma ‖hₜ‖;
  - similitud media entre tokens y rango efectivo (si los tokens colapsan a lo mismo);
  - **logit lens**: aplicar `final_norm` + `out_head` al estado de cada iteración y ver desde cuándo la respuesta es correcta.
- **Esperado:** la iteración en la que aparece la respuesta crece con el nivel. Repetir los mismos pesos no hace explotar la norma ni colapsar los tokens (si lo hace, es un hallazgo).

### H9 · Qué pasa dentro del bucle (atención)

- **Qué:** las iteraciones del looped reutilizan patrones de atención parecidos entre sí, y la atención hacia los números de la pregunta se concentra a medida que avanza el bucle.
- **Cómo:** por iteración y cabeza:
  - entropía normalizada de cada fila;
  - divergencia de Jensen‑Shannon entre iteraciones consecutivas;
  - fracción de atención que va a los dígitos de la pregunta desde la posición de la respuesta.

  En el tipo B desordenado, comprobar si la atención "salta" entre las frases que forman la cadena.
- **Esperado:** divergencia baja entre iteraciones del looped y alta entre capas del GPT; atención a los números creciente.

### Métricas

- **Principal:** exactitud exacta por nivel (porcentaje de respuestas con el número exacto), por fichero, junto a las referencias de la sección 7.3.
- **Secundaria:** entropía cruzada media sobre los tokens de la respuesta, con teacher forcing.
- **Coste:** parámetros, FLOPs por token y tiempo de entrenamiento.

---

## 10. Notas para cada parte del equipo

### Parte 2 · Modelos y entorno

- `vocab_size = 97` y `context_length = 512` vienen de aquí; impórtalos de `src/tokenizer.py` en lugar de escribirlos a mano.
- Los embeddings posicionales son absolutos: por eso los ficheros de evaluación nunca superan la longitud máxima de `train`.
- `src/calibrate.py` lleva una copia del `GPTModel` del notebook para no depender de vosotros. Cuando exista `src/model.py`, la calibración puede pasar a usarlo.
- La atención explícita del notebook (matriz de pesos materializada) no cabe en 4 GB con batch 64; la copia de `calibrate.py` usa `F.scaled_dot_product_attention` y GELU fusionada (mismo resultado, diferencia ~7e‑7). Para entrenar, `model.py` debería ofrecer esa versión; para el análisis de atención (H9) hace falta la explícita, que devuelve los pesos.

### Parte 3 · Entrenamiento y evaluación

- Construye el texto con `format_example` y el prompt de evaluación con `format_prompt`, no a mano: así nadie desalinea el formato.
- La máscara de la pérdida: `-100` desde el principio hasta el último carácter de `"Respuesta: "` incluido; objetivos reales en los dígitos de la respuesta y en el `<eos>`. `EncodedSplit.batch` en `src/calibrate.py` es una implementación de referencia, comprobada (sobreajusta 16 ejemplos hasta pérdida 0,001 y 100 % de exactitud).
- **No rellenes por la izquierda** para generar en batch: con posiciones absolutas desplaza el texto. Agrupa por longitud de prompt, genera uno a uno, o usa el atajo de teacher forcing de la sección 6.1.
- Reporta la exactitud separada por `level`, y también por `type` y `shuffled` (H6). Incluye `test_tpl` en la evaluación (H7).
- Para desarrollar antes de tener el dataset completo, usa `data/sample/`.
- Presupuesto de referencia: 42 000 pasos con batch 64 (sección 8.1). Con 3 semillas, usad la media y no interpretéis diferencias de menos de 3–4 puntos por nivel.

### Parte 4 · Análisis y artículo

- Para H8 y H9 usa la posición del último carácter de `"Respuesta: "`: es donde el modelo tiene que haber calculado el resultado antes de escribir el primer dígito.
- Para la masa de atención sobre los números, las posiciones de los dígitos de la pregunta se obtienen directamente del texto (cada dígito es un token).
- En las tablas, incluye las referencias de la sección 7.3, y al comentar los niveles 1–2 ten presente la sección 7.2.
- La sección de datos del artículo puede salir casi entera de este documento (en particular la sección 3, la decisión del rango, va como decisión de diseño y como limitación) y de `data/data_card.md`.

---

## 11. Limitaciones conocidas

- **Rango cerrado de valores** (1–99): es una decisión de diseño tomada con la calibración; sus consecuencias están en la sección 3.
- **Respuestas pequeñas frecuentes:** el 39 % de las respuestas tiene una cifra y el 1 es la más frecuente (7,4 %).
- **Aritmética repetida en los niveles bajos:** ver sección 7.2.
- **La longitud sigue correlacionada con el nivel**, aunque se solapa mucho (sección 6.3). Un modelo podría usarla como pista débil de la dificultad.
- **El tope de longitud en evaluación** hace que los niveles 5–6 sean sobre todo compactos (86–91 %).
- **Las cadenas con muchas sumas y restas** siguen siendo en parte "sumables" sin seguir el orden; exigir una operación multiplicativa y prohibir las que se anulan lo reduce, pero no lo elimina.
- **Las redacciones son finitas** (65 de operaciones en total). `test_tpl` mide hasta qué punto importa (en la calibración, mucho: sección 8.1).
- **No hay tipo C** (dos cadenas que se combinan al final) ni distractores. Quedan como ampliación si sobra tiempo.

---

## 12. Glosario

- **Nivel:** número de operaciones necesarias para llegar a la respuesta.
- **Traza (`trace`):** la solución como lista evaluable de izquierda a derecha.
- **OOD (*out of distribution*):** datos de un tipo no visto al entrenar; aquí, niveles 5–6.
- **Redacción reservada:** forma de expresar una operación que solo aparece en `test_tpl`.
- **Teacher forcing:** evaluar dando al modelo la respuesta correcta como entrada en cada paso, en lugar de su propia predicción.
- **Logit lens:** aplicar la cabeza de salida al estado intermedio de una capa o iteración para ver qué predeciría el modelo si parara ahí.
- **Calibración:** entrenar un modelo de referencia para comprobar que la dificultad deja margen para comparar arquitecturas.
