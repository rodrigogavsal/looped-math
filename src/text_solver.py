"""Resolvedor independiente: calcula la respuesta leyendo SOLO el texto del problema.

No usa la traza ni importa nada del generador: sus patrones están escritos a partir
del significado de las frases en castellano. Si su respuesta coincide con `answer`
en todos los problemas, el enunciado expresa de verdad lo que dice la traza
(incluidas las frases inversas y las desordenadas del tipo B).

Uso:
    python -m src.text_solver data/train.jsonl data/val.jsonl ...
Lo usa también tests/test_data.py.
"""
import json
import re
import sys
from collections import Counter
from fractions import Fraction

FRAC = {"tercera": 3, "cuarta": 4}
MULT = {"doble": 2, "triple": 3, "cuádruple": 4, "dos": 2, "tres": 3}


def split_question(q):
    """Separa el cuerpo de la pregunta final (quitando 'Entonces,' / 'Al final,' delante del '¿')."""
    body, _, question = q.rpartition("¿")
    body = re.sub(r"\s*(Entonces|Al final),\s*$", "", body)
    return body, question


# ---------------------------------------------------------------------------- tipo A
# Cada patrón reconoce una operación en el texto; se aplican en el orden en que aparecen.
A_PATTERNS = [
    (r"\ble regala (\d+)", "+"),
    (r"\b(?:gana|compra|recibe|encuentra|consigue|meten|ponen|mete|añade)\s+(\d+)", "+"),
    (r"\ble da (\d+)", "-"),
    (r"\b(?:regala|pierde|pierden|vende|sacan|saca|quitan|lleva)\s+(\d+)", "-"),
    (r"\b(?:duplica|duplican)\b|tant[ao]s \w+ como ya (?:tenía|había)", "*2"),
    (r"\b(?:triplica|triplican)\b|\bmultiplican? por tres\b", "*3"),
    (r"\bla mitad\b", "/2"),
    (r"\b(?:en|hacen) (\d) (?:montones|grupos) iguales", "/k"),
    (r"\bla (tercera|cuarta) parte", "/frac"),
]


def solve_a(q):
    """Devuelve (respuesta, nº de operaciones reconocidas)."""
    body, _ = split_question(q)
    m = re.search(r"(?:tiene|hay|contiene|con|guarda) (\d+) ", body)
    v = Fraction(int(m.group(1)))
    rest = body[m.end():].lower()
    events = []
    for pat, op in A_PATTERNS:
        for mm in re.finditer(pat, rest):
            events.append((mm.start(), mm.end(), op, mm.group(1) if mm.groups() else None))
    events.sort()
    n_ops, last_end = 0, -1
    for start, end, op, g in events:
        if start < last_end:  # solapado con un patrón ya aplicado
            continue
        last_end = end
        n_ops += 1
        if op == "+":
            v += int(g)
        elif op == "-":
            v -= int(g)
        elif op == "*2":
            v *= 2
        elif op == "*3":
            v *= 3
        elif op == "/2":
            v /= 2
        elif op == "/k":
            v /= int(g)
        elif op == "/frac":
            v /= FRAC[g]
    return v, n_ops


# ---------------------------------------------------------------------------- tipo B
_N = r"([A-ZÁÉÍÓÚÑ]\w+)"
B_PATTERNS = [
    (rf"^(?:Sabemos que )?{_N} tiene (?:exactamente )?(\d+) \w+$", "base"),
    (rf"^{_N} tiene tant\w+ \w+ como {_N} y (\d+) más$", "tantos"),
    (rf"^A {_N} le faltan? (\d+) \w+ para tener tant\w+ como {_N}$", "faltan"),
    (rf"^{_N} tiene (\d+)(?: \w+)? más que {_N}$", "mas"),
    (rf"^{_N} tiene (\d+)(?: \w+)? menos que {_N}$", "menos"),
    (rf"^{_N} tiene el (doble|triple|cuádruple)(?: de \w+)? que {_N}$", "mult"),
    (rf"^{_N} tiene (dos|tres) veces lo que tiene {_N}$", "veces"),
    (rf"^{_N} tiene la mitad(?: de \w+)? que {_N}$", "mitad"),
    (rf"^{_N} tiene la (tercera|cuarta) parte de l[ao]s \w+ de {_N}$", "parte"),
]


def solve_b(q):
    """Devuelve (respuesta, nº de relaciones). Resuelve el sistema propagando valores conocidos."""
    body, question = split_question(q)
    target = re.findall(r"[A-ZÁÉÍÓÚÑ]\w+", question)[-1]
    known, eqs = {}, []  # cada ecuación: X = Y + c  o  X = Y * c
    for sent in [s.strip() for s in body.split(".") if s.strip()]:
        if "amigos" in sent:  # frase de ambientación, sin números
            continue
        for pat, kind in B_PATTERNS:
            m = re.match(pat, sent)
            if not m:
                continue
            g = m.groups()
            if kind == "base":
                known[g[0]] = Fraction(int(g[1]))
            elif kind == "tantos":
                eqs.append((g[0], g[1], "+", int(g[2])))
            elif kind == "faltan":
                eqs.append((g[0], g[2], "+", -int(g[1])))
            elif kind == "mas":
                eqs.append((g[0], g[2], "+", int(g[1])))
            elif kind == "menos":
                eqs.append((g[0], g[2], "+", -int(g[1])))
            elif kind in ("mult", "veces"):
                eqs.append((g[0], g[2], "*", Fraction(MULT[g[1]])))
            elif kind == "mitad":
                eqs.append((g[0], g[1], "*", Fraction(1, 2)))
            elif kind == "parte":
                eqs.append((g[0], g[2], "*", Fraction(1, FRAC[g[1]])))
            break
        else:
            raise ValueError(f"frase no reconocida: {sent!r}")
    changed = True
    while changed:  # cada ecuación se puede usar en los dos sentidos
        changed = False
        for X, Y, op, c in eqs:
            if Y in known and X not in known:
                known[X] = known[Y] + c if op == "+" else known[Y] * c
                changed = True
            elif X in known and Y not in known:
                known[Y] = known[X] - c if op == "+" else known[X] / c
                changed = True
    return known[target], len(eqs)


def solve(row):
    """(respuesta, nº de operaciones) leídas del texto de una fila del JSONL."""
    return solve_a(row["question"]) if row["type"] == "A" else solve_b(row["question"])


def check_rows(rows):
    """Lista de discrepancias (id, motivo) entre el texto y los campos answer/level."""
    bad = []
    for r in rows:
        try:
            v, n_ops = solve(r)
        except Exception as e:  # noqa: BLE001
            bad.append((r["id"], f"error: {e}"))
            continue
        if v != r["answer"]:
            bad.append((r["id"], f"texto={v} answer={r['answer']}"))
        if n_ops != r["level"]:
            bad.append((r["id"], f"operaciones en el texto={n_ops} level={r['level']}"))
    return bad


def main(paths):
    ok = True
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        bad = check_rows(rows)
        types = Counter(r["type"] for r in rows)
        print(f"{path}: {len(rows)} problemas (A={types['A']}, B={types['B']}), discrepancias={len(bad)}")
        for b in bad[:10]:
            print("   ", b)
        ok &= not bad
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
