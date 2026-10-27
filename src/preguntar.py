"""Pregunta a uno o varios modelos entrenados con tus propios problemas (modo inferencia).

Uso (desde la raíz del repositorio, con el entorno activado):
    python -m src.preguntar --runs gpt_L8_d256_s0 looped_N8_d256_s0
        -> modo interactivo: escribe un problema y pulsa Enter (línea vacía para salir)
    python -m src.preguntar --runs gpt_L8_d256_s0 --q "Ana tiene 12 canicas. Gana 7 canicas. ¿Cuántas canicas tiene Ana?"
    python -m src.preguntar --runs looped_N8_d256_s0 --loops 10 --q "..."     # looped con otro nº de loops

Usa exactamente el mismo formato de prompt y la misma generación greedy que la evaluación
(src/evaluate.py): "Pregunta: <texto>\\nRespuesta: " y hasta 6 tokens o <eos>.
Avisa si el texto tiene caracteres fuera del vocabulario (el modelo los ve como <unk>).
"""
import argparse

import torch

from src.evaluate import extract_number, generate_greedy_batch
from src.model import build_model
from src.tokenizer import get_tokenizer


def load(run, dev):
    ckpt = torch.load(f"checkpoints/{run}.pt", map_location=dev)
    model = build_model(ckpt["cfg"]).to(dev)
    model.load_state_dict(ckpt["model_state"])
    return model.eval()


def ask(model, tok, dev, question, loops=None):
    ids = tok.encode(f"Pregunta: {question}\nRespuesta: ")
    out = generate_greedy_batch(model, torch.tensor([ids], device=dev), max_new_tokens=6,
                                eos_id=tok.eos_id, n_loops=loops)[0]
    text = tok.decode(out, skip_special=True)
    return text, extract_number(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--q", default=None, help="un problema; sin --q, modo interactivo")
    ap.add_argument("--loops", type=int, default=None, help="solo looped: nº de loops en inferencia")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    tok = get_tokenizer()
    models = {r: load(r, dev) for r in a.runs}
    unk = tok.char_to_id.get("<unk>") if hasattr(tok, "char_to_id") else None

    def run_one(q):
        bad = sorted({c for c in q if c not in tok.char_to_id}) if hasattr(tok, "char_to_id") else []
        if bad:
            print(f"  (aviso: caracteres fuera del vocabulario, el modelo los ve como <unk>: {''.join(bad)!r})")
        for r, m in models.items():
            text, num = ask(m, tok, dev, q, a.loops)
            print(f"  {r:22s} -> {text!r}")

    if a.q:
        run_one(a.q)
        return
    print("Escribe un problema (línea vacía para salir).")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            break
        run_one(q)


if __name__ == "__main__":
    main()
