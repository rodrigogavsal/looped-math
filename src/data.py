"""Dataset, DataLoader y utilidades de datos según la Sección 2.2 y Parte 3 del contrato común.

Maneja:
- Lectura de ficheros JSON Lines (train.jsonl, val.jsonl, test_id.jsonl, test_ood.jsonl)
- Construcción del texto: "Pregunta: " + question + "\nRespuesta: " + str(answer) + "<eos>"
- Creación de pares (x, y) donde la pérdida se evalúa solo en los caracteres de la respuesta y en <eos> (-100 en el resto)
- Collate dinámico (padding hasta la longitud máxima del batch)
- Generador de dataset de juguete para poder trabajar sin esperar los datos reales de Pau
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import CharTokenizer, get_tokenizer

IGNORE_INDEX = -100


class MathDataset(Dataset):
    """Dataset para problemas matemáticos en JSONL con enmascaramiento causal según contrato."""

    def __init__(
        self,
        data_source: Union[str, List[Dict[str, Any]]],
        tokenizer: Optional[CharTokenizer] = None,
        context_length: int = 512,
    ):
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()
        self.context_length = context_length

        if isinstance(data_source, str):
            if not os.path.exists(data_source):
                raise FileNotFoundError(f"No se encontró el fichero de datos: {data_source}")
            self.items = []
            with open(data_source, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.items.append(json.loads(line))
        else:
            self.items = list(data_source)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        question = item["question"]
        answer_str = str(item["answer"])
        level = int(item.get("level", 1))

        # Texto del prompt que ve el modelo antes de generar la respuesta
        prompt_text = f"Pregunta: {question}\nRespuesta: "

        # Tokens
        prompt_tokens = self.tokenizer.encode(prompt_text)
        answer_tokens = self.tokenizer.encode(answer_str)
        full_tokens = prompt_tokens + answer_tokens + [self.tokenizer.eos_id]

        if len(full_tokens) > self.context_length:
            # En caso de superar context_length (no debería ocurrir según contrato), recortamos
            full_tokens = full_tokens[: self.context_length]

        # x: todos los tokens excepto el último
        # y: todos los tokens desplazados en 1
        x = full_tokens[:-1]
        y = list(full_tokens[1:])

        # Enmascaramiento: la pérdida se calcula SOLO sobre la respuesta y el <eos>.
        # Los índices de prompt en y van desde 0 hasta len(prompt_tokens) - 2.
        # En el índice len(prompt_tokens) - 1 de x (último token del prompt, el espacio),
        # el objetivo en y es el primer carácter de la respuesta. Ese sí se calcula.
        prompt_len = len(prompt_tokens)
        for i in range(max(0, prompt_len - 1)):
            if i < len(y):
                y[i] = IGNORE_INDEX

        return {
            "x": torch.tensor(x, dtype=torch.long),
            "y": torch.tensor(y, dtype=torch.long),
            "level": level,
            "id": item.get("id", f"sample_{idx}"),
            "question": question,
            "answer": item["answer"],
            "prompt_text": prompt_text,
            "prompt_len": prompt_len,
        }


class MathCollate:
    """Collate con relleno dinámico (padding) a la longitud máxima del batch."""

    def __init__(self, pad_id: int = 0, ignore_index: int = IGNORE_INDEX):
        self.pad_id = pad_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(item["x"]) for item in batch)
        batch_size = len(batch)

        x_padded = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        y_padded = torch.full((batch_size, max_len), self.ignore_index, dtype=torch.long)
        levels = torch.tensor([item["level"] for item in batch], dtype=torch.long)

        ids = []
        questions = []
        answers = []
        prompt_texts = []
        prompt_lens = []

        for b_idx, item in enumerate(batch):
            seq_len = len(item["x"])
            x_padded[b_idx, :seq_len] = item["x"]
            y_padded[b_idx, :seq_len] = item["y"]
            ids.append(item["id"])
            questions.append(item["question"])
            answers.append(item["answer"])
            prompt_texts.append(item["prompt_text"])
            prompt_lens.append(item["prompt_len"])

        return {
            "x": x_padded,
            "y": y_padded,
            "levels": levels,
            "ids": ids,
            "questions": questions,
            "answers": answers,
            "prompt_texts": prompt_texts,
            "prompt_lens": prompt_lens,
        }


def create_dataloader(
    data_source: Union[str, List[Dict[str, Any]]],
    tokenizer: Optional[CharTokenizer] = None,
    batch_size: int = 64,
    shuffle: bool = True,
    context_length: int = 512,
    num_workers: int = 0,
) -> DataLoader:
    """Crea un DataLoader con MathDataset y MathCollate."""
    dataset = MathDataset(data_source, tokenizer=tokenizer, context_length=context_length)
    tok = dataset.tokenizer
    collate_fn = MathCollate(pad_id=tok.pad_id, ignore_index=IGNORE_INDEX)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Generador de datos sintéticos de juguete (Toy Dataset)
# Para desbloquear el pipeline inmediatamente sin esperar a Pau (Parte 1).
# ---------------------------------------------------------------------------

NAMES = ["Ana", "Luis", "Carlos", "María", "Pedro", "Lucía", "Juan", "Elena"]
OBJECTS = ["canicas", "manzanas", "caramelos", "libros", "cromos", "pegatinas", "monedas"]


def generate_toy_problem(level: int, prob_id: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """Genera un problema de matemáticas en castellano con N operaciones (nivel 1 a 6)."""
    if seed is not None:
        random.seed(seed)

    name = random.choice(NAMES)
    obj = random.choice(OBJECTS)

    # El nivel indica el número de operaciones (1 a 6)
    num_ops = level

    current_val = random.randint(10, 40)
    trace: List[Union[int, str]] = [current_val]
    story_parts = [f"{name} tiene {current_val} {obj}."]

    ops = ["+", "-"] if num_ops > 1 else ["+"]
    connectors = ["Luego,", "Después,", "Más tarde,", "Al final,"]

    for step in range(num_ops):
        op = random.choice(ops)
        conn = connectors[step % len(connectors)]

        if op == "+":
            val = random.randint(1, 20)
            current_val += val
            trace.extend(["+", val])
            story_parts.append(f"{conn} gana {val} {obj}.")
        else:
            # Asegurar que no sea negativo
            max_take = max(1, current_val - 1)
            val = random.randint(1, min(15, max_take))
            current_val -= val
            trace.extend(["-", val])
            other_name = random.choice([n for n in NAMES if n != name])
            story_parts.append(f"{conn} le da {val} a {other_name}.")

    story_parts.append(f"¿Cuántas {obj} tiene {name}?")
    question = " ".join(story_parts)

    return {
        "id": prob_id,
        "level": level,
        "question": question,
        "answer": current_val,
        "trace": trace,
        "template": f"toy_level_{level}",
    }


def create_toy_dataset_splits(
    output_dir: str = "data",
    samples_per_level_train: int = 250,
    samples_per_level_val: int = 50,
    samples_per_level_test: int = 50,
    seed: int = 42,
) -> Dict[str, str]:
    """Genera las 4 particiones en JSONL según la Sección 2.2 del contrato:

    - train.jsonl (niveles 1-4)
    - val.jsonl (niveles 1-4)
    - test_id.jsonl (niveles 1-4)
    - test_ood.jsonl (niveles 5-6)
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    splits = {
        "train": (output_dir + "/train.jsonl", [1, 2, 3, 4], samples_per_level_train),
        "val": (output_dir + "/val.jsonl", [1, 2, 3, 4], samples_per_level_val),
        "test_id": (output_dir + "/test_id.jsonl", [1, 2, 3, 4], samples_per_level_test),
        "test_ood": (output_dir + "/test_ood.jsonl", [5, 6], samples_per_level_test),
    }

    result_paths = {}
    counter = 0

    for split_name, (file_path, levels, n_samples) in splits.items():
        with open(file_path, "w", encoding="utf-8") as f:
            for lvl in levels:
                for i in range(n_samples):
                    counter += 1
                    prob_id = f"L{lvl}-{counter:06d}"
                    prob = generate_toy_problem(lvl, prob_id)
                    f.write(json.dumps(prob, ensure_ascii=False) + "\n")
        result_paths[split_name] = file_path

    return result_paths


if __name__ == "__main__":
    paths = create_toy_dataset_splits()
    print("Datos de juguete generados correctamente en:", paths)
