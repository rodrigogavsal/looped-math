# Parche sobre las Partes 2 y 3

Cinco cambios para poder lanzar la matriz de experimentos. Lo demás del código no se ha tocado.
Tests: 61 pasan (`pytest -q`), entre ellos los 30 originales.

## 1. Memoria de GPU (`src/model.py`, `src/train.py`, `src/benchmark.py`, `configs/`)

Con la atención explícita, GPT de 4 capas con batch 64 no cabía en una GPU de 4 GB (lo vimos en la calibración).

- **Atención:** con `return_attn=False` (entrenar y evaluar) se usa `F.scaled_dot_product_attention`, que no guarda la matriz T×T. Con `return_attn=True` se sigue usando la versión explícita del notebook, que devuelve los pesos para el análisis. Mismo resultado (`tests/test_memory.py`).
- **GELU:** `F.gelu(x, approximate="tanh")`, misma fórmula que el notebook pero fusionada.
- **`grad_accum: N`:** cada paso se hace en N trozos de `batch_size/N` ejemplos. La pérdida se normaliza por el total de tokens del batch completo, así que el gradiente es el mismo que con batch 64, y los trozos son exactamente los mismos 64 ejemplos. Puesto a 2 en `gpt_L8`, `gpt_L12`, `looped_N8` y `looped_N12`.
- **`grad_checkpoint: true`:** recalcula las activaciones de cada bloque en el backward. Mismo resultado, ~30 % más lento. Solo si `grad_accum` no basta.
- **`python -m src.benchmark`:** si un modelo no cabe con batch 64, prueba con 2 y 4 trozos y luego con checkpointing, y dice cuál cabe. Por defecto mide `seq_len` 411 (el ejemplo más largo del dataset) y estima horas para 42 000 pasos.

Memoria de activaciones medida (batch 64, 411 tokens, bf16; sin contar pesos, optimizador ni el contexto de CUDA): ~0,35 GB por capa o loop con la atención eficiente. Profundidad 4 ≈ 1,5 GB; 8 ≈ 2,9 GB; 12 ≈ 4,3 GB. Con `grad_accum: 2`, la mitad. Con checkpointing, ~0,03 GB por capa.

## 2. Pasos (`configs/base.yaml`)

`steps` y `total_steps` pasan de 10 000 a 42 000, el presupuesto validado en la calibración (`docs/DATASET.md`, sección 8.1). Con ~10 000 pasos los niveles 3–4 aún no han despegado. `eval_every` pasa a 1000.

## 3. Tokenizador único y Parte 1 integrada

- `src/tokenizer.py` es el de la Parte 1: los IDs del contrato (`.` = 81, `,` = 82; antes estaban intercambiados) y `format_prompt`, `format_example`, `MAX_LEN`, etc. Mantiene la API que usa vuestro código: `get_tokenizer()`, `vocab`, `char_to_id`, `PAD_ID`… y la normalización de `*` a `×`.
- Añadidos de la Parte 1: `src/generate_data.py`, `src/text_solver.py`, `src/calibrate.py`, `src/eval_templates.py`, `docs/DATASET.md`, `data/sample/` y sus tests (`tests/test_dataset.py`).
- Test nuevo que fija los IDs del contrato (`tests/test_tokenizer.py`).

## 4. Evaluación (`src/evaluate.py`, `src/train.py`, `src/run.py`, `looped.py`)

- Se evalúa también **`test_tpl`** (H7).
- Cada fichero de test da, además de `acc_by_level`, **`acc_total`** y **`acc_by_type`**: `A`, `B ordenado` y `B desordenado` (H6).
- Barrido de loops en **`test_ood`** además de en `test_id` (H5). `loops_sweep` sigue siendo el de `test_id`, como antes; el nuevo es `loops_sweep_ood`.
- **`results/<run>_preds.jsonl`:** una línea por problema con `split`, `id`, `level`, `type`, `template`, `answer`, `pred`, `generated` y `correct`, para la Parte 4.

## 5. Lanzador (`src/run.py`)

- `--runs gpt_L1` lanza solo `gpt_L1_d256`, ya no también `gpt_L12`. `--runs gpt_L4` ya no lanza también el opcional `gpt_L4_d128`: este solo se lanza por su nombre completo o con `--all --include-optional`.
- Faltaba `import torch`: `--device` daba error.
- Si faltan datos, el lanzador para con un mensaje que explica cómo generarlos. Antes generaba datos de juguete en silencio; ahora solo lo hace con `--toy-data`.
- Opciones nuevas: `--grad-accum N` y `--grad-checkpoint`.
