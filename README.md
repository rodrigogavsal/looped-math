# Looped Transformer para Razonamiento Aritmético (`looped-math`)

Repositorio del proyecto de **Procesamiento del Lenguaje Natural I (PLN I)** sobre **Looped Transformers** entrenados desde cero para resolver problemas matemáticos en castellano de dificultad creciente.

---

## 1. Estructura del repositorio

```text
looped-math/
├── README.md                  # Instrucciones de instalación, uso y registro de hardware
├── requirements.txt           # Dependencias de Python
├── check_env.py               # Verificación de PyTorch, CUDA, GPU, VRAM y soporte bf16
├── configs/
│   ├── base.yaml              # Hiperparámetros por defecto
│   └── experiments.yaml       # Matriz de configuraciones y semillas
├── data/                      # Datasets JSONL generados (ignorados en git salvo data_card.md)
│   └── data_card.md           # Estadísticas y documentación del dataset
├── src/
│   ├── __init__.py
│   ├── model.py               # GPTModel, LoopedGPT, build_model, parámetros y FLOPs
│   ├── benchmark.py           # Benchmark de memoria VRAM, velocidad y tablas LaTeX/MD
│   └── analysis/              # Métricas de representación, atención y figuras
├── tests/                     # Suite de tests con pytest (test_model.py)
├── notebooks/                 # Cuadernos con la implementación original
│   ├── base.ipynb             # Implementación original de GPT-2 base e instrumentación
│   └── loop.ipynb             # Implementación original de LoopedGPT y experimentos interactivos
├── checkpoints/               # Pesos entrenados (.pt, ignorado en git)
├── results/                   # Métricas y curvas (.json, ignorado en git)
├── figures/                   # Gráficas y tablas generadas (ignorado en git)
└── paper/                     # Plantilla LaTeX del artículo
```

### 1.1 Descripción de cada carpeta y origen del código

Este repositorio organiza tanto la exploración interactiva como la base de código modular para el proyecto:

* **`notebooks/` (Implementación original de los modelos y exploración interactiva)**:
  Fue el punto de partida y la **implementación original** de las arquitecturas:
  - `base.ipynb`: Implementación original del modelo GPT estándar (atención causal multi-cabeza, Pre-LayerNorm, bloques Transformer, `GPTModel` e instrumentación de estados ocultos y atención).
  - `loop.ipynb`: Implementación original de la recurrencia en profundidad (`LoopedGPT`), herencia de bloques compartidos, factoría unificada `build_model`, funciones de desenrollado (`unroll_to_gpt`) y conversión de pesos (`looped_from_gpt`), importando las definiciones base dinámicamente.
  > **Nota sobre el diseño**: Los *notebooks* sirvieron como entorno de prototipado y validación inicial de los modelos. Posteriormente, **todo el código de las arquitecturas definido en ellos fue extraído y consolidado en `src/model.py`**, dotándolo de tipado estático estricto, manejo robusto de excepciones y desacoplándolo del entorno interactivo.

* **`src/` (Código fuente modular de Python)**:
  Contiene los módulos Python limpios, reutilizables y preparados para ejecución por scripts:
  - `model.py`: Módulo donde se integró la implementación definitiva de los modelos extraídos de los cuadernos (`GPTModel`, `LoopedGPT`, `TransformerBlock`, `MultiHeadAttention`, etc.), junto al cálculo analítico de parámetros y FLOPs por token (`flops_per_token`) y utilidades de conversión bidireccional.
  - `benchmark.py`: Medición rigurosa de uso de memoria VRAM, velocidad por paso de entrenamiento y exportación automatizada de tablas en Markdown y LaTeX para el informe.
  - `analysis/`: Utilidades para el análisis de representaciones internas, mapas de atención y similitud de representaciones.

* **`configs/` (Configuración de hiperparámetros)**:
  Ficheros YAML para garantizar reproducibilidad sin modificar código fuente:
  - `base.yaml`: Hiperparámetros generales por defecto (arquitectura, optimizador AdamW, planificación de tasa de aprendizaje, longitud de contexto).
  - `experiments.yaml`: Definición formal de la matriz experimental completa (variaciones de capas $L$, iteraciones $N$, dimensión oculta $d$ y semillas).

* **`tests/` (Suite de pruebas automatizadas)**:
  - `test_model.py`: Batería de 20 tests unitarios con `pytest` que verifican la corrección matemática de `src/model.py`: equivalencia exacta entre `LoopedGPT` desenrollado y `GPTModel`, invarianza de parámetros al variar $N$, causalidad estricta de la máscara y modos de inicialización.

* **`data/` (Datos y especificaciones)**:
  Directorio donde se generan y leen los conjuntos de datos de problemas aritméticos en formato JSONL (entrenamiento, validación y test ID/OOD), acompañado de la ficha técnica `data_card.md`.

* **`checkpoints/` (Pesos de modelos)**:
  Directorio para almacenar los checkpoints de pesos entrenados (`.pt`) de cada experimento y semilla.

* **`results/` (Métricas y logs experimentales)**:
  Archivos `.json` con curvas de pérdida, precisión por nivel de dificultad, evaluaciones de inferencia y resultados de benchmarks.

* **`figures/` (Gráficas y tablas generadas)**:
  Tablas formateadas (`model_params_flops.md`, `model_params_flops.tex`, `benchmark_vram_speed.md`) y gráficas generadas listas para la memoria del trabajo.

* **`paper/` (Artículo científico)**:
  Fuentes LaTeX y recursos bibliográficos para la redacción final del artículo o memoria académica.

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

## 3. Verificación rápida y Benchmark de Hardware

Para ejecutar los tests de corrección del modelo:

```bash
pytest -v
```

Para generar la tabla de parámetros y FLOPs (`figures/model_params_flops.md` y `.tex`) y medir la memoria VRAM y velocidad por paso en tu máquina (`figures/benchmark_vram_speed.md`):

```bash
python -m src.benchmark
```

---

## 4. Registro de máquinas del equipo

Ejecutad `python check_env.py` en cada ordenador y anotad los datos en la siguiente tabla:

| Miembro | SO | PyTorch | GPU | VRAM (GB) | Soporta `bf16` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Máquina 1 | Windows 11 (AMD64) | 2.14.0+cpu | CPU (CUDA no disponible) | - | No |
| Máquina 2 | *Pendiente* | - | - | - | - |
| Máquina 3 | *Pendiente* | - | - | - | - |
| Máquina 4 | *Pendiente* | - | - | - | - |

---

## 5. Tabla de Parámetros y FLOPs de la Matriz Experimental

Valores calculados con `vocab_size = 97` y `context_length = 512` (`python -m src.benchmark --skip-benchmark`):

| Modelo | Arq. | $d$ | Cabezas | Bloques ($L$) | Loops ($N$) | Prof. efectiva | Parámetros | Parámetros (M) | FLOPs / token (M) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt_L1_d256` | GPT | 256 | 8 | 1 | 1 | 1 | 970,240 | 0.970 | 2.15 |
| `gpt_L2_d256` | GPT | 256 | 8 | 2 | 1 | 2 | 1,759,232 | 1.759 | 4.24 |
| `gpt_L4_d256` | GPT | 256 | 8 | 4 | 1 | 4 | 3,337,216 | 3.337 | 8.44 |
| `gpt_L8_d256` | GPT | 256 | 8 | 8 | 1 | 8 | 6,493,184 | 6.493 | 16.83 |
| `gpt_L12_d256` | GPT | 256 | 8 | 12 | 1 | 12 | 9,649,152 | 9.649 | 25.22 |
| `looped_N2_d256` | Looped | 256 | 8 | 1 | 2 | 2 | 970,240 | 0.970 | 4.24 |
| `looped_N4_d256` | Looped | 256 | 8 | 1 | 4 | 4 | 970,240 | 0.970 | 8.44 |
| `looped_N8_d256` | Looped | 256 | 8 | 1 | 8 | 8 | 970,240 | 0.970 | 16.83 |
| `looped_N12_d256` | Looped | 256 | 8 | 1 | 12 | 12 | 970,240 | 0.970 | 25.22 |
| `gpt_L4_d128` *(opc.)* | GPT | 128 | 4 | 4 | 1 | 4 | 882,176 | 0.882 | 2.65 |

