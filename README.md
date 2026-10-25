# Looped Transformer para Razonamiento Aritmético (`looped-math`)

Repositorio del proyecto de **Procesamiento del Lenguaje Natural I (PLN I)** sobre **Looped Transformers** entrenados desde cero para resolver problemas matemáticos en castellano de dificultad creciente.

---

## 1. Estructura del repositorio

```text
looped-math/
├── README.md                  # Documentación del proyecto, guía de instalación y arquitectura
├── requirements.txt           # Dependencias del entorno Python
├── pytest.ini                 # Configuración del test runner pytest (pythonpath = .)
├── check_env.py               # Verificación de PyTorch, CUDA, GPU, VRAM y soporte bf16
├── looped.py                  # CLI unificado para entrenamiento, evaluación, demos y sanity checks
├── configs/                   # Configuraciones reproducibles en formato YAML
│   ├── base.yaml              # Hiperparámetros generales y de entrenamiento por defecto
│   └── experiments.yaml       # Matriz experimental de configuraciones (L, N, d) y semillas
├── data/                      # Conjuntos de datos JSONL generados (ignorado en git)
│   └── data_card.md           # Ficha técnica, formato y estadísticas del dataset aritmético
├── notebooks/                 # Cuadernos interactivos (IMPLEMENTACIÓN ORIGINAL DE LOS MODELOS)
│   ├── base.ipynb             # Implementación original de GPT-2 base, bloques y componentes
│   └── loop.ipynb             # Implementación original de LoopedGPT, entrenamiento y evaluación
├── src/                       # Paquete de código modular de Python extraído de los notebooks
│   ├── __init__.py
│   ├── tokenizer.py           # Tokenizador por caracteres (CharTokenizer) con vocabulario fijo
│   ├── data.py                # Dataset (MathDataset) y DataLoader con máscara causal sobre la pregunta
│   ├── model.py               # GPTModel, LoopedGPT, build_model, parámetros y cálculo de FLOPs
│   ├── train.py               # Bucle de entrenamiento con AMP, scheduler warmup+coseno y checkpoints
│   ├── evaluate.py            # Evaluación por teacher forcing, generación greedy y barrido de loops
│   ├── run.py                 # Orquestador secuencial y reanudable de la matriz de experimentos
│   ├── benchmark.py           # Benchmark de hardware (VRAM, velocidad) y tablas LaTeX/MD
│   └── analysis/              # Métricas y utilidades para mapas de atención y representaciones
├── tests/                     # Suite de pruebas automatizadas con pytest (30 tests)
│   ├── __init__.py
│   ├── test_tokenizer.py      # Tests unitarios del tokenizador (caracteres, OOV, padding)
│   ├── test_data.py           # Tests del Dataset, DataLoader y enmascaramiento con IGNORE_INDEX
│   ├── test_model.py          # 20 tests de equivalencia (Looped vs GPT unrolled, causalidad, logits)
│   ├── test_train.py          # Tests del optimizador, scheduler de LR y sobreajuste
│   └── test_evaluate.py       # Tests de extracción de números, teacher forcing y métricas
├── checkpoints/               # Checkpoints de pesos entrenados (.pt, ignorado en git)
├── results/                   # Métricas, historiales de pérdida y evaluaciones (.json, ignorado en git)
├── figures/                   # Gráficas generadas y tablas comparativas (ignorado en git)
└── paper/                     # Fuentes y plantilla LaTeX del artículo científico
```

### 1.1 Descripción detallada de cada carpeta y origen del código

El proyecto nació a partir de cuadernos interactivos de investigación y posteriormente evolucionó hacia una base de código modular estructurada para experimentación a gran escala:

* **`notebooks/` (Implementación original de los modelos y entorno exploratorio)**:
  Esta carpeta contiene el origen y la **implementación fundacional** de todo el proyecto:
  - `base.ipynb`: Implementación original del modelo GPT estándar desde cero (atención multi-cabeza causal, Pre-LayerNorm, bloques Transformer, `GPTModel` e instrumentación de estados ocultos y atención), basada en la metodología de *LLMs from Scratch*. También incluye el prototipo inicial del tokenizador por caracteres y el dataset básico.
  - `loop.ipynb`: Implementación original de la recurrencia en profundidad (`LoopedGPT`), la técnica de pesos compartidos mediante `@property shared_blocks`, la factoría `build_model`, el desenrollado en GPT estándar (`unroll_to_gpt`), las funciones de conversión de pesos (`looped_from_gpt`), así como los primeros prototipos del bucle de entrenamiento y la evaluación desglosada por niveles.
  > **Origen del código**: Los notebooks permitieron prototipar, depurar interactivamente y verificar visualmente la arquitectura. **A partir de ellos, todo el código definido originalmente en los notebooks se extrajo, refactorizó y trasladó a los módulos de `src/`**, dotándolo de tipado estático estricto (`typing`), manejo formal de excepciones, desacoplamiento y compatibilidad total con scripts y pipelines reproducibles.

* **`src/` (Paquete modular de Python)**:
  Contiene la implementación limpia y desacoplada de todo el pipeline, extraída de los cuadernos:
  - `model.py`: Módulo donde se integraron y consolidaron las arquitecturas de red neuronal (`GPTModel`, `LoopedGPT`, `TransformerBlock`, `MultiHeadAttention`, etc.), junto al cálculo analítico de parámetros y FLOPs por token (`flops_per_token`) y utilidades de conversión bidireccional entre GPT y Looped.
  - `tokenizer.py`: Implementación formal de `CharTokenizer`, que asigna enteros deterministas a un vocabulario fijo de 97 caracteres (dígitos, operadores, espacios, signos y delimitadores) y define `IGNORE_INDEX = -100`.
  - `data.py`: Clase `MathDataset` con filtrado por nivel de dificultad, `collate_math_batch` con relleno automático (*right-padding*) y enmascaramiento estricto del prompt (se asigna `IGNORE_INDEX` a la pregunta para calcular la pérdida únicamente sobre los tokens de la respuesta), junto a `create_dataloader` y `create_toy_dataset_splits`.
  - `train.py`: Pipeline completo de entrenamiento (`train_run`) con optimizador AdamW con *weight decay* desacoplado, recorte de norma de gradientes a 1.0, planificador de tasa de aprendizaje con calentamiento lineal y decaimiento coseno (`lr_schedule`), soporte de precisión mixta automática (AMP en `bfloat16` o `float16`), guardado y reanudación de checkpoints y prueba de sobreajuste rápido (`sanity_check_overfit`).
  - `evaluate.py`: Métricas rigurosas de rendimiento: evaluación por *teacher forcing* por nivel (`evaluate_teacher_forcing_by_level`), evaluación autorregresiva por generación *greedy* y extracción regex del entero predicho (`evaluate_generation_by_level`), barrido de número de loops en inferencia (`evaluate_loops_sweep`) y formateadores de visualización.
  - `run.py`: Script orquestador para ejecutar la matriz experimental completa de `configs/experiments.yaml` de manera secuencial, desatendida y reanudable si se interrumpe.
  - `benchmark.py`: Medición empírica de pico de VRAM y velocidad por paso de entrenamiento en el hardware local, exportando tablas automáticas en Markdown y LaTeX para la memoria.
  - `analysis/`: Submódulo destinado al análisis de representaciones internas, similitud CKA y mapas de atención.

* **`tests/` (Suite de pruebas automatizadas)**:
  Batería de 30 tests con `pytest` que garantizan la integridad de todo el pipeline y certifican que la refactorización a `src/` replica con exactitud milimétrica la lógica de los cuadernos:
  - `test_model.py`: 20 tests que validan la equivalencia numérica de logits entre LoopedGPT y GPT desenrollado, causalidad estricta de la atención, invarianza del número de parámetros ante variaciones de $N$ y modos de conversión.
  - `test_tokenizer.py`: Validación de codificación y decodificación exacta, caracteres fuera de vocabulario y casos especiales.
  - `test_data.py`: Validación del dataset, dataloader y correcta aplicación de la máscara de pérdida `IGNORE_INDEX` sobre el prompt.
  - `test_train.py`: Validación del optimizador, del scheduler de aprendizaje y del test de sobreajuste.
  - `test_evaluate.py`: Validación de la extracción regex de respuestas, evaluación *teacher forcing* y barrido de loops.

* **`configs/` (Configuración declarativa de experimentos)**:
  Archivos YAML que desacoplan los parámetros del código para máxima reproducibilidad:
  - `base.yaml`: Configuración base y por defecto (dimensiones, optimizador, contexto, directorios).
  - `experiments.yaml`: Definición de la matriz de modelos a comparar (L1, L2, L4, L8, L12 vs N2, N4, N8, N12) y sus semillas aleatorias.

* **`data/` (Almacenamiento de datos y documentación)**:
  Directorio donde residen los datos sintéticos generados en formato JSONL y su ficha descriptiva `data_card.md`.

* **`checkpoints/` (Pesos de modelos)**:
  Guarda los estados de los modelos entrenados (`.pt`) incluyendo pesos y estado del optimizador para reanudación o inferencia.

* **`results/` (Resultados cuantitativos)**:
  Almacena los historiales de entrenamiento (`loss`), métricas de evaluación por nivel y benchmarks en formato JSON estructurado.

* **`figures/` (Gráficas y tablas para la publicación)**:
  Salida de gráficas de evolución de pérdida, precisión por nivel, mapas de calor de atención y tablas formateadas para Markdown y LaTeX.

* **`paper/` (Memoria académica y artículo)**:
  Directorio para los manuscritos LaTeX, bibliografía BibTeX y recursos del artículo final del proyecto.

* **Archivos clave en la raíz**:
  - `looped.py`: Interfaz de línea de comandos (CLI) principal y de conveniencia para interactuar rápidamente con el proyecto (`--sanity-check`, `--train`, `--eval`, `--demo`).
  - `check_env.py`: Script de diagnóstico del entorno de cómputo local (GPU, VRAM, versión de PyTorch, soporte bf16).
  - `pytest.ini`: Configuración global que fija `pythonpath = .` y `testpaths = tests`.
  - `requirements.txt`: Lista de dependencias fijas del proyecto.

---

## 2. Instalación

### En Windows (PowerShell)

```powershell
# 1. Crear y activar entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Instalar PyTorch con soporte CUDA (ajustar cu121 / cu124 según versión de driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Instalar el resto de dependencias
pip install -r requirements.txt

# 4. Comprobar el entorno y GPU
python check_env.py
```

### En Ubuntu / Linux (Bash)

```bash
# 1. Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar PyTorch con soporte CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Instalar el resto de dependencias
pip install -r requirements.txt

# 4. Comprobar el entorno y GPU
python check_env.py
```

---

## 3. Verificación rápida, Tests y CLI

### 3.1 Suite de pruebas automatizadas
Para ejecutar los 30 tests unitarios que verifican el tokenizador, datos, modelos, entrenamiento y evaluación:

```bash
pytest -v
```

### 3.2 Prueba de sobreajuste rápido (*Sanity Check*)
Para verificar que el pipeline de datos, modelo y optimizador converge sobreajustando 32 ejemplos sintéticos hasta pérdida cero:

```bash
python looped.py --sanity-check
```

### 3.3 Interfaz de comandos CLI (`looped.py`)
El script principal permite entrenar, evaluar y probar modelos fácilmente:

```bash
# Entrenar una configuración específica (e.g. Looped N=4, d=256 con semilla 0)
python looped.py --train --run looped_N4_d256 --seed 0

# Evaluar un checkpoint generado en test ID y OOD desglosado por nivel
python looped.py --eval --checkpoint checkpoints/looped_N4_d256_s0.pt

# Generar 10 ejemplos interactivos de muestra con un modelo entrenado
python looped.py --demo --checkpoint checkpoints/looped_N4_d256_s0.pt
```

### 3.4 Benchmark de Hardware y coste computacional
Para generar la tabla de parámetros y FLOPs (`figures/model_params_flops.md` y `.tex`) y medir la memoria VRAM y velocidad por paso en tu máquina (`figures/benchmark_vram_speed.md`):

```bash
python -m src.benchmark
```

### 3.5 Lanzador de la matriz de experimentos completa
Para ejecutar de forma reanudable y secuencial todos los modelos de `configs/experiments.yaml`:

```bash
python -m src.run
```