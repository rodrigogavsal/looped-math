"""Generador del dataset de problemas de matemáticas en castellano (Parte 1).

Uso (desde la raíz del repositorio):
    python -m src.generate_data --out data                 # dataset completo
    python -m src.generate_data --out data/sample --sample # muestra: 50 problemas por nivel y fichero

Ver docs/DATASET.md para el diseño completo. Resumen:

- Nivel = número de operaciones encadenadas (1-6).
- Dos tipos de problema, al 50 % en cada nivel:
    A · Secuencia: una cantidad que cambia paso a paso (una persona o un recipiente).
    B · Cadena de comparaciones: varias personas definidas unas respecto a otras;
        se pregunta por la última. En la mitad, las frases están desordenadas.
- Operaciones: sumar, restar, el doble, el triple, la mitad y la tercera/cuarta
  parte (solo divisiones exactas). Operandos de 1 a 20; todas las cantidades
  (iniciales, intermedias y finales) entre 1 y 99.
- A partir del nivel 2, cada problema tiene al menos una operación multiplicativa
  (doble, triple, mitad, parte), para que el orden importe y no baste con sumar
  todos los números con su signo.
- Nunca hay dos operaciones seguidas que se anulen (doble y luego mitad, +5 y luego -5).
- Los ficheros se escriben siempre con finales de línea "\n" (idénticos en Windows y Linux).
- Cada operación tiene varias redacciones; una por operación se reserva y SOLO
  aparece en test_tpl.jsonl (test de plantillas no vistas).

Ficheros (contrato común, sección 2.2):
    train.jsonl     niveles 1-4
    val.jsonl       niveles 1-4
    test_id.jsonl   niveles 1-4
    test_tpl.jsonl  niveles 1-4, cada problema usa al menos una redacción reservada
    test_ood.jsonl  niveles 5-6
  Ningún fichero de evaluación tiene ejemplos más largos que el más largo de train.
    data_card.md    estadísticas
"""
import argparse
import json
import os
import random
import re
import statistics
from collections import Counter, defaultdict

try:  # ejecutado como módulo (python -m src.generate_data) o importado desde src
    from .tokenizer import CharTokenizer, MAX_LEN, example_length, format_example
except ImportError:  # ejecutado como script suelto
    from tokenizer import CharTokenizer, MAX_LEN, example_length, format_example

# ============================================================================ vocabulario de dominio
NAMES = [
    "Ana", "Luis", "Marta", "Pablo", "Lucía", "Sara", "Carlos", "Elena", "Jorge", "Nuria",
    "Iván", "Raúl", "Inés", "Óscar", "Lola", "Teo", "Sofía", "Hugo", "Clara", "Dani",
    "Eva", "Bea", "Ciro", "Dora", "Edu", "Fran", "Gala", "Irene", "Javier", "Laura",
    "Mario", "Noelia", "Pedro", "Rocío", "Sergio", "Tania", "Víctor", "Alba", "Héctor",
    "Julia", "Álvaro", "Ángela", "Mateo", "Paula", "Andrés", "Celia", "Nico", "Vera",
]

# (singular, plural, género)
OBJECTS = [
    ("canica", "canicas", "f"), ("cromo", "cromos", "m"), ("manzana", "manzanas", "f"),
    ("libro", "libros", "m"), ("caramelo", "caramelos", "m"), ("galleta", "galletas", "f"),
    ("pegatina", "pegatinas", "f"), ("sello", "sellos", "m"), ("carta", "cartas", "f"),
    ("globo", "globos", "m"), ("naranja", "naranjas", "f"), ("flor", "flores", "f"),
    ("moneda", "monedas", "f"), ("cuaderno", "cuadernos", "m"), ("piedra", "piedras", "f"),
    ("lápiz", "lápices", "m"), ("bombón", "bombones", "m"), ("tornillo", "tornillos", "m"),
    ("concha", "conchas", "f"), ("postal", "postales", "f"), ("figura", "figuras", "f"),
    ("rotulador", "rotuladores", "m"), ("botón", "botones", "m"), ("peonza", "peonzas", "f"),
]

# Solo recipientes femeninos: evita las contracciones "del" / "al".
CONTAINERS = ["caja", "cesta", "bolsa", "mochila", "lata", "bandeja"]

FAMILY = [
    "su abuela", "su abuelo", "su hermana", "su hermano", "su prima", "su primo",
    "su madre", "su padre", "su tía", "su tío", "una amiga", "un amigo", "una vecina", "un vecino",
]

CONNECTORS = ["luego", "después", "más tarde", "entonces"]
FRAC = {3: "tercera", 4: "cuarta"}
MULT = {3: "triple", 4: "cuádruple"}

# Rango por defecto fijado con la calibración (docs/DATASET.md, sección 8.1):
# con 1-50 / 1-200 un GPT de 4 capas no despega (nivel 4 ~6 %); con 1-20 / 1-99 sí.
MIN_VAL, MAX_VAL, MAX_OPERAND = 1, 99, 20
OP_WEIGHTS = {"add": 0.26, "sub": 0.26, "mul2": 0.12, "mul3": 0.08, "half": 0.16, "div": 0.12}
ADDITIVE = {"add", "sub"}

# Probabilidad de redacción "verbose" por nivel: los niveles bajos se redactan más
# largo y los altos más compacto, para que la longitud no delate el nivel.
VERBOSE_P = {1: 0.8, 2: 0.65, 3: 0.5, 4: 0.35, 5: 0.15, 6: 0.1}

# ============================================================================ plantillas
# Cada redacción: (id, estilo, reservada, texto, flags)
#   estilo: "c" compacta / "v" verbose
#   reservada: True -> solo aparece en test_tpl
#   flags: "verb" = la cláusula empieza por verbo (se le puede anteponer el sujeto)
#          "conn" = ya lleva un complemento temporal (no se le añade conector)
# Las cláusulas de operaciones del tipo A van en minúscula y sin punto final:
# el ensamblador añade conectores, une con "y", pone mayúscula y punto.

A_PERSON = {
    "start": [
        ("ps_tiene", "c", False, "{N} tiene {q}.", ()),
        ("ps_empieza", "c", False, "{N} empieza el día con {q}.", ()),
        ("ps_principio", "v", False, "Al principio, {N} tiene {q}.", ()),
        ("ps_guarda", "v", False, "{N} guarda {q} en su habitación.", ()),
    ],
    "filler": ["A {N} le gusta coleccionar {o}.", "{N} siempre lleva la cuenta de sus {o}."],
    "ops": {
        "add": [
            ("pa_gana", "c", False, "gana {q} más", ("verb",)),
            ("pa_compra", "c", False, "compra {q}", ("verb",)),
            ("pa_recibe", "c", True, "recibe {q} de regalo", ("verb",)),
            ("pa_regalo", "v", False, "{fam} le regala {q}", ()),
            ("pa_encuentra", "v", False, "encuentra {q} más en un cajón", ("verb",)),
            ("pa_colegio", "v", False, "en el colegio consigue {q} más", ("conn",)),
        ],
        "sub": [
            ("ps_regala", "c", False, "regala {q}", ("verb",)),
            ("ps_pierde", "c", False, "pierde {q}", ("verb",)),
            ("ps_da", "v", False, "le da {q} a {fam}", ("verb",)),
            ("ps_vende", "v", True, "vende {q} a un compañero", ("verb",)),
            ("ps_parque", "v", False, "en el parque {se_pierden} {q}", ("conn",)),
        ],
        "mul2": [
            ("pm2_duplica", "c", False, "duplica sus {o}", ("verb",)),
            ("pm2_sedup", "v", True, "al día siguiente el número de {o} que tiene se duplica", ("conn",)),
            ("pm2_tantos", "v", False, "{fam} le da {tantos} {o} como ya tenía", ()),
        ],
        "mul3": [
            ("pm3_triplica", "c", False, "triplica sus {o}", ("verb",)),
            ("pm3_mult", "v", False, "multiplica por tres sus {o}", ("verb",)),
            ("pm3_setrip", "v", True, "el número de {o} que tiene se triplica", ()),
        ],
        "half": [
            ("ph_regala", "c", False, "regala la mitad de sus {o}", ("verb",)),
            ("ph_pierde", "v", True, "pierde la mitad de sus {o} durante una excursión", ("verb",)),
            ("ph_queda", "c", False, "se queda solo con la mitad de sus {o}", ("verb",)),
        ],
        "div": [
            ("pd_montones", "v", False, "reparte sus {o} en {k} montones iguales y se queda con uno", ("verb",)),
            ("pd_parte", "c", True, "se queda solo con la {frac} parte de sus {o}", ("verb",)),
            ("pd_grupos", "c", False, "divide sus {o} en {k} grupos iguales y guarda solo uno", ("verb",)),
        ],
    },
    "question": [
        ("pq_ahora", "¿{Cuant} {o} tiene {N} ahora?"),
        ("pq_final", "¿{Cuant} {o} tiene {N} al final?"),
        ("pq_queda", "¿Con {cuant} {o} se queda {N}?"),
    ],
}

A_CONTAINER = {
    "start": [
        ("cs_hay", "c", False, "En una {cont} hay {q}.", ()),
        ("cs_contiene", "v", False, "Una {cont} contiene {q}.", ()),
        ("cs_empezar", "v", False, "Al empezar el día, en una {cont} hay {q}.", ()),
    ],
    "filler": ["La {cont} está encima de la mesa.", "Nadie sabe quién ha dejado allí la {cont}."],
    "ops": {
        "add": [
            ("ca_meten", "c", False, "meten {q} más", ()),
            ("ca_ponen", "c", False, "ponen {q} más", ()),
            ("ca_anade", "v", False, "alguien añade {q} a la {cont}", ()),
            ("ca_nina", "v", True, "una niña mete {q} más", ()),
        ],
        "sub": [
            ("cs_sacan", "c", False, "sacan {q}", ()),
            ("cs_quitan", "c", False, "quitan {q}", ()),
            ("cs_lleva", "v", False, "alguien se lleva {q} de la {cont}", ()),
            ("cs_nino", "v", True, "un niño saca {q}", ()),
        ],
        "mul2": [
            ("cm2_c", "c", False, "duplican {art} {o}", ()),
            ("cm2_dup", "v", False, "el número de {o} de la {cont} se duplica", ()),
            ("cm2_tantos", "v", True, "meten {tantos} {o} como ya había", ()),
        ],
        "mul3": [
            ("cm3_trip", "v", False, "el número de {o} de la {cont} se triplica", ()),
            ("cm3_mult", "c", False, "multiplican por tres {art} {o}", ()),
            ("cm3_c", "c", True, "triplican {art} {o}", ()),
        ],
        "half": [
            ("ch_sacan", "c", False, "sacan la mitad de {art} {o}", ()),
            ("ch_quitan", "c", False, "quitan la mitad", ()),
            ("ch_lleva", "v", True, "alguien se lleva la mitad de {art} {o}", ()),
        ],
        "div": [
            ("cd_grupos", "v", False, "reparten {art} {o} en {k} grupos iguales y dejan solo uno en la {cont}", ()),
            ("cd_montones", "c", False, "hacen {k} montones iguales y dejan solo uno", ()),
            ("cd_parte", "c", True, "dejan solo la {frac} parte de {art} {o}", ()),
        ],
    },
    "question": [
        ("cq_ahora", "¿{Cuant} {o} hay ahora en la {cont}?"),
        ("cq_quedan", "¿{Cuant} {o} quedan en la {cont}?"),
        ("cq_final", "Al final, ¿{cuant} {o} hay en la {cont}?"),
    ],
}

# Tipo B: frases completas. {A} = persona ya definida, {B} = persona nueva (B = A op k).
B_CHAIN = {
    "start": [
        ("bs_tiene", "c", False, "{A} tiene {q}.", ()),
        ("bs_sabemos", "v", False, "Sabemos que {A} tiene {q}.", ()),
        ("bs_exacto", "v", False, "{A} tiene exactamente {q}.", ()),
    ],
    "filler": ["Varios amigos comparan sus {o}.", "Unos amigos cuentan {art} {o} que tiene cada uno."],
    "ops": {
        "add": [
            ("ba_mas", "c", False, "{B} tiene {q} más que {A}.", ()),
            ("ba_k", "c", False, "{B} tiene {k} más que {A}.", ()),
            ("ba_inv", "v", False, "{A} tiene {q} menos que {B}.", ()),
            ("ba_tantos", "v", True, "{B} tiene {tantos} {o} como {A} y {k} más.", ()),
        ],
        "sub": [
            ("bs_menos", "c", False, "{B} tiene {q} menos que {A}.", ()),
            ("bs_inv", "v", False, "{A} tiene {q} más que {B}.", ()),
            ("bs_faltan", "v", False, "A {B} le {faltan} {q} para tener {tantos} como {A}.", ()),
            ("bs_k", "c", True, "{B} tiene {k} menos que {A}.", ()),
        ],
        "mul2": [
            ("bm2_doble", "c", False, "{B} tiene el doble de {o} que {A}.", ()),
            ("bm2_k", "c", False, "{B} tiene el doble que {A}.", ()),
            ("bm2_inv", "v", False, "{A} tiene la mitad de {o} que {B}.", ()),
            ("bm2_veces", "v", True, "{B} tiene dos veces lo que tiene {A}.", ()),
        ],
        "mul3": [
            ("bm3_triple", "c", False, "{B} tiene el triple de {o} que {A}.", ()),
            ("bm3_veces", "v", False, "{B} tiene tres veces lo que tiene {A}.", ()),
            ("bm3_inv", "v", False, "{A} tiene la tercera parte de {art} {o} de {B}.", ()),
            ("bm3_k", "c", True, "{B} tiene el triple que {A}.", ()),
        ],
        "half": [
            ("bh_mitad", "c", False, "{B} tiene la mitad de {o} que {A}.", ()),
            ("bh_k", "c", False, "{B} tiene la mitad que {A}.", ()),
            ("bh_inv", "v", True, "{A} tiene el doble de {o} que {B}.", ()),
        ],
        "div": [
            ("bd_parte", "v", False, "{B} tiene la {frac} parte de {art} {o} de {A}.", ()),
            ("bd_inv", "c", False, "{A} tiene el {mult} que {B}.", ()),
            ("bd_inv2", "v", True, "{A} tiene el {mult} de {o} que {B}.", ()),
        ],
    },
    "question": [
        ("bq_tiene", "¿{Cuant} {o} tiene {Z}?"),
        ("bq_entonces", "Entonces, ¿{cuant} {o} tiene {Z}?"),
        ("bq_numero", "¿Cuál es el número de {o} de {Z}?"),
    ],
}

HELD_IDS = sorted({p[0] for tpl in (A_PERSON, A_CONTAINER, B_CHAIN)
                   for pool in tpl["ops"].values() for p in pool if p[2]})


# ============================================================================ aritmética
def eval_trace(trace):
    """Evalúa una traza [v0, op, arg, op, arg, ...] de izquierda a derecha.

    Devuelve (resultado, lista de valores intermedios incluido v0). Lanza
    ValueError si una división no es exacta.
    """
    v = trace[0]
    values = [v]
    for sym, arg in zip(trace[1::2], trace[2::2]):
        if sym == "+":
            v = v + arg
        elif sym == "-":
            v = v - arg
        elif sym == "*":
            v = v * arg
        elif sym == "/":
            if v % arg:
                raise ValueError(f"división no exacta {v}/{arg}")
            v = v // arg
        else:
            raise ValueError(sym)
        values.append(v)
    return v, values


def _try_op(op, v, rng):
    """Devuelve (símbolo, argumento, nuevo valor) si la operación es válida desde v, o None."""
    if op == "add":
        hi = min(MAX_OPERAND, MAX_VAL - v)
        if hi < 1:
            return None
        k = rng.randint(1, hi)
        return "+", k, v + k
    if op == "sub":
        hi = min(MAX_OPERAND, v - MIN_VAL)
        if hi < 1:
            return None
        k = rng.randint(1, hi)
        return "-", k, v - k
    if op == "mul2":
        return ("*", 2, 2 * v) if 2 * v <= MAX_VAL else None
    if op == "mul3":
        return ("*", 3, 3 * v) if 3 * v <= MAX_VAL else None
    if op == "half":
        return ("/", 2, v // 2) if v % 2 == 0 and v // 2 >= MIN_VAL else None
    if op == "div":
        ks = [k for k in (3, 4) if v % k == 0 and v // k >= MIN_VAL]
        if not ks:
            return None
        k = rng.choice(ks)
        return "/", k, v // k
    raise ValueError(op)


def is_inverse(prev, cur):
    """True si `cur` deshace justo la operación anterior `prev` (pares (símbolo, argumento)).

    Doble seguido de mitad, triple seguido de tercera parte, +k seguido de -k, y al revés.
    Estos pares se prohíben: dejan el valor igual que antes, así que el problema tendría
    menos pasos reales que su nivel y la respuesta saldría copiando un número del texto.
    """
    if prev is None:
        return False
    (ps, pa), (cs, ca) = prev, cur
    return pa == ca and {ps, cs} in ({"*", "/"}, {"+", "-"})


def _try_op_not_inverse(op, v, rng, prev, tries=4):
    """Como _try_op, pero descarta resultados que deshacen la operación anterior."""
    for _ in range(tries):
        r = _try_op(op, v, rng)
        if r is None:
            return None
        if not is_inverse(prev, r[:2]):
            return r
    return None


def sample_chain(level, rng):
    """Valor inicial y `level` operaciones válidas.

    Nivel >= 2: al menos una operación multiplicativa, y nunca dos operaciones seguidas
    que se anulen (ver is_inverse).
    """
    ops, weights = list(OP_WEIGHTS), list(OP_WEIGHTS.values())
    while True:
        v0 = rng.randint(1, MAX_OPERAND)
        v, steps, prev = v0, [], None
        for _ in range(level):
            feasible = [(op, w, r) for op, w in zip(ops, weights)
                        if (r := _try_op_not_inverse(op, v, rng, prev))]
            if not feasible:
                break
            op, _, (sym, arg, v) = rng.choices(feasible, weights=[f[1] for f in feasible])[0]
            steps.append((op, sym, arg))
            prev = (sym, arg)
        if len(steps) < level:
            continue
        if level >= 2 and all(op in ADDITIVE for op, _, _ in steps):
            continue
        return v0, steps, v


# ============================================================================ texto
def _qty(n, obj):
    sing, plur, _ = obj
    return f"{n} {sing if n == 1 else plur}"


def _cap(s):
    return s[:1].upper() + s[1:]


def _base_ctx(obj):
    fem = obj[2] == "f"
    return {"o": obj[1], "Cuant": "Cuántas" if fem else "Cuántos", "cuant": "cuántas" if fem else "cuántos",
            "tantos": "tantas" if fem else "tantos", "art": "las" if fem else "los"}


def _step_ctx(base, sym, arg, obj, rng):
    ctx = dict(base)
    ctx["fam"] = rng.choice(FAMILY)
    if sym in "+-":
        ctx["k"] = arg
        ctx["q"] = _qty(arg, obj)
        ctx["se_pierden"] = "se le pierde" if arg == 1 else "se le pierden"
        ctx["faltan"] = "falta" if arg == 1 else "faltan"
    elif sym == "/" and arg in FRAC:
        ctx["k"] = arg
        ctx["frac"] = FRAC[arg]
        ctx["mult"] = MULT[arg]
    return ctx


def _pick(pool, verbose, rng, held_mode, avoid=None):
    """Elige una redacción. held_mode: 'exclude' (normal), 'only' (forzar reservada) o 'any'.

    `avoid`: id de la redacción anterior; se evita repetirla seguida si hay alternativa.
    """
    if held_mode == "exclude":
        cands = [p for p in pool if not p[2]]
    elif held_mode == "only":
        return rng.choice([p for p in pool if p[2]])
    else:
        cands = list(pool)
    cands = [p for p in cands if p[0] != avoid] or cands
    styled = [p for p in cands if p[1] == ("v" if verbose else "c")]
    return rng.choice(styled or cands)


def _held_modes(level, tpl_mode, rng):
    if not tpl_mode:
        return ["exclude"] * level
    forced = rng.randrange(level)
    return ["only" if i == forced else "any" for i in range(level)]


def make_type_a(level, rng, verbose, tpl_mode=False):
    frame_name = rng.choice(["persona", "recipiente"])
    tpl = A_PERSON if frame_name == "persona" else A_CONTAINER
    obj = rng.choice(OBJECTS)
    name = rng.choice(NAMES)
    cont = rng.choice(CONTAINERS)
    v0, steps, answer = sample_chain(level, rng)
    base = _base_ctx(obj)
    base.update(N=name, cont=cont)

    sentences, ids = [], []
    start = _pick(tpl["start"], verbose, rng, "any")
    ids.append(start[0])
    start_txt = start[3].format(**{**base, "q": _qty(v0, obj)})
    if verbose and rng.random() < 0.5:
        filler = rng.choice(tpl["filler"]).format(**base)
        # persona: el relleno puede ir antes; recipiente: siempre después de presentarlo
        sentences += [filler, start_txt] if frame_name == "persona" else [start_txt, filler]
    else:
        sentences.append(start_txt)

    p_conn, p_subj, p_merge = (0.7, 0.4, 0.3) if verbose else (0.35, 0.15, 0.2)
    conns_left = CONNECTORS[:]  # cada conector se usa como mucho una vez por problema
    rng.shuffle(conns_left)
    parts, merged_prev, prev_id = [], False, None
    for i, ((op, sym, arg), hm) in enumerate(zip(steps, _held_modes(level, tpl_mode, rng))):
        ph = _pick(tpl["ops"][op], verbose, rng, hm, avoid=prev_id)
        prev_id = ph[0]
        ids.append(ph[0])
        clause = ph[3].format(**_step_ctx(base, sym, arg, obj, rng))
        flags = ph[4]
        merge = (bool(parts) and not merged_prev and " y " not in parts[-1] and " y " not in clause
                 and ph[0] != ids[-2] and rng.random() < p_merge)
        if frame_name == "persona" and "verb" in flags and not merge and rng.random() < p_subj:
            clause = f"{name} {clause}"
        if "conn" not in flags and i > 0 and rng.random() < p_conn:
            if i == level - 1 and level > 2 and rng.random() < 0.5:
                clause = f"al final {clause}"
            elif conns_left:
                clause = f"{conns_left.pop()} {clause}"
        if merge:
            parts[-1] = f"{parts[-1]} y {clause}"
            merged_prev = True
        else:
            parts.append(clause)
            merged_prev = False
    sentences += [_cap(p) + "." for p in parts]

    q = rng.choice(tpl["question"])
    ids.append(q[0])
    sentences.append(q[1].format(**base))
    trace = [v0] + [x for _, sym, arg in steps for x in (sym, arg)]
    return {"type": "A", "frame": frame_name, "shuffled": False, "question": " ".join(sentences),
            "answer": answer, "trace": trace, "template": "A-" + frame_name + "|" + "|".join(ids)}


def make_type_b(level, rng, verbose, shuffle, tpl_mode=False):
    obj = rng.choice(OBJECTS)
    names = rng.sample(NAMES, level + 1)
    v0, steps, answer = sample_chain(level, rng)
    base = _base_ctx(obj)

    ids = []
    start = _pick(B_CHAIN["start"], verbose, rng, "any")
    ids.append(start[0])
    facts = [start[3].format(**{**base, "A": names[0], "q": _qty(v0, obj)})]
    prev_id = None
    for i, ((op, sym, arg), hm) in enumerate(zip(steps, _held_modes(level, tpl_mode, rng))):
        ph = _pick(B_CHAIN["ops"][op], verbose, rng, hm, avoid=prev_id)
        prev_id = ph[0]
        ids.append(ph[0])
        ctx = _step_ctx(base, sym, arg, obj, rng)
        ctx.update(A=names[i], B=names[i + 1])
        facts.append(ph[3].format(**ctx))

    order = list(range(len(facts)))
    if shuffle:
        while order == sorted(order):
            rng.shuffle(order)
    sentences = []
    if verbose and rng.random() < 0.5:
        sentences.append(rng.choice(B_CHAIN["filler"]).format(**base))
    sentences += [facts[j] for j in order]
    q = rng.choice(B_CHAIN["question"])
    ids.append(q[0])
    sentences.append(q[1].format(**{**base, "Z": names[-1]}))
    trace = [v0] + [x for _, sym, arg in steps for x in (sym, arg)]
    return {"type": "B", "frame": "cadena", "shuffled": shuffle, "question": " ".join(sentences),
            "answer": answer, "trace": trace, "template": "B|" + "|".join(ids)}


def make_problem(level, ptype, rng, tpl_mode=False):
    verbose = rng.random() < VERBOSE_P[level]
    if ptype == "A":
        p = make_type_a(level, rng, verbose, tpl_mode)
    else:
        p = make_type_b(level, rng, verbose, shuffle=rng.random() < 0.5, tpl_mode=tpl_mode)
    p["style"] = "verbose" if verbose else "compact"
    return p


# ============================================================================ particiones
SPLITS = [  # (nombre, niveles, clave del tamaño, modo plantillas reservadas)
    ("train", (1, 2, 3, 4), "n_train", False),
    ("val", (1, 2, 3, 4), "n_val", False),
    ("test_id", (1, 2, 3, 4), "n_test", False),
    ("test_tpl", (1, 2, 3, 4), "n_test", True),
    ("test_ood", (5, 6), "n_test", False),
]


def generate_split(name, levels, n_per_level, tpl_mode, rng, seen, max_len):
    """Genera n_per_level problemas por nivel, sin repetir preguntas de `seen`."""
    tok = CharTokenizer()
    rows = []
    for level in levels:
        made = 0
        while made < n_per_level:
            ptype = "A" if made % 2 == 0 else "B"
            p = make_problem(level, ptype, rng, tpl_mode)
            if p["question"] in seen:
                continue
            if example_length(p["question"], p["answer"]) > max_len:
                continue
            if not tok.is_supported(format_example(p["question"], p["answer"])):
                raise ValueError(f"caracteres fuera del vocabulario: "
                                 f"{tok.unsupported_chars(p['question'])} en {p['question']!r}")
            seen.add(p["question"])
            row = {"id": f"{name}-L{level}-{made:06d}", "level": level, "type": p["type"],
                   "shuffled": p["shuffled"], "style": p["style"], "question": p["question"],
                   "answer": p["answer"], "trace": p["trace"], "template": p["template"]}
            rows.append(row)
            made += 1
    rng.shuffle(rows)
    return rows


def generate_dataset(out_dir, n_train=50_000, n_val=1_000, n_test=2_000, seed=1234, write_card=True,
                     verbose=True):
    """Genera los cinco ficheros JSONL (y la data card) en out_dir. Tamaños por nivel."""
    os.makedirs(out_dir, exist_ok=True)
    sizes = {"n_train": n_train, "n_val": n_val, "n_test": n_test}
    seen = set()
    all_rows = {}
    train_max = MAX_LEN
    for i, (name, levels, size_key, tpl_mode) in enumerate(SPLITS):
        rng = random.Random(seed * 1000 + i)
        # Ningún ejemplo de evaluación supera la longitud máxima vista en train: con embeddings
        # posicionales absolutos, las posiciones nunca entrenadas romperían la comparación.
        max_len = MAX_LEN if name == "train" else train_max
        rows = generate_split(name, levels, sizes[size_key], tpl_mode, rng, seen, max_len)
        if name == "train":
            train_max = max(example_length(r["question"], r["answer"]) for r in rows)
        all_rows[name] = rows
        path = os.path.join(out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if verbose:
            print(f"{name:9s} {len(rows):7d} problemas -> {path}")
    if write_card:
        card = data_card(all_rows, seed, sizes)
        with open(os.path.join(out_dir, "data_card.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write(card)
    return all_rows


# ============================================================================ data card
def _numbers(text):
    return [int(x) for x in re.findall(r"\d+", text)]


def _baselines_section(all_rows):
    """Exactitud de heurísticas que no razonan y solapamiento de trazas con train."""
    train = all_rows.get("train", [])
    if not train:
        return []
    L = ["## Referencias: heurísticas que no razonan\n",
         "Exactitud que se obtendría sin resolver nada. Cualquier modelo tiene que superarlas con claridad.\n",
         "| Fichero | Nivel | Siempre la respuesta más frecuente de train | Primer número del texto | Último número del texto |",
         "|---|---|---|---|---|"]
    majority = Counter(r["answer"] for r in train).most_common(1)[0][0]
    for name in ("test_id", "test_ood"):
        by_level = defaultdict(list)
        for r in all_rows.get(name, []):
            by_level[r["level"]].append(r)
        for level in sorted(by_level):
            rs = by_level[level]
            acc = lambda f: f"{100 * sum(f(r) == r['answer'] for r in rs) / len(rs):.1f} %"  # noqa: E731
            L.append(f"| {name} | {level} | {acc(lambda r: majority)} | {acc(lambda r: _numbers(r['question'])[0])} | "
                     f"{acc(lambda r: _numbers(r['question'])[-1])} |")
    L += ["", f"La respuesta más frecuente de train es {majority}.", "",
          "## Trazas de evaluación que ya aparecen en train\n",
          "Porcentaje de problemas cuya traza (mismos números y operaciones) aparece en train con otro texto. "
          "No es fuga de datos (el texto es distinto), pero en los niveles bajos la aritmética ya se ha visto.\n",
          "| Fichero | " + " | ".join(f"Nivel {l}" for l in range(1, 7)) + " |", "|---|" + "---|" * 6]
    seen = {tuple(r["trace"]) for r in train}
    for name in ("test_id", "test_tpl", "test_ood"):
        by_level = defaultdict(list)
        for r in all_rows.get(name, []):
            by_level[r["level"]].append(tuple(r["trace"]) in seen)
        cells = [f"{100 * sum(by_level[l]) / len(by_level[l]):.1f} %" if by_level.get(l) else "—" for l in range(1, 7)]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")
    return L


def data_card(all_rows, seed, sizes):
    L = []
    L.append("# Data card\n")
    L.append(f"Generado con `python -m src.generate_data` (semilla {seed}, tamaños por nivel {sizes}, "
             f"operando máximo {MAX_OPERAND}, valor máximo {MAX_VAL}).\n")
    L.append(f"Redacciones reservadas para `test_tpl` ({len(HELD_IDS)}): " + ", ".join(f"`{h}`" for h in HELD_IDS) + "\n")
    L.append("## Tamaños y longitudes (tokens por ejemplo, incluido `<eos>`)\n")
    L.append("| Fichero | Nivel | Problemas | Tipo A | B desordenados | Verbose | Long. media | Long. máx. | Respuesta media | Respuesta máx. |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, rows in all_rows.items():
        by_level = defaultdict(list)
        for r in rows:
            by_level[r["level"]].append(r)
        for level in sorted(by_level):
            rs = by_level[level]
            lens = [example_length(r["question"], r["answer"]) for r in rs]
            b = [r for r in rs if r["type"] == "B"]
            L.append(f"| {name} | {level} | {len(rs)} | {sum(r['type'] == 'A' for r in rs) / len(rs):.0%} | "
                     f"{(sum(r['shuffled'] for r in b) / len(b)) if b else 0:.0%} | "
                     f"{sum(r['style'] == 'verbose' for r in rs) / len(rs):.0%} | {statistics.mean(lens):.0f} | "
                     f"{max(lens)} | {statistics.mean(r['answer'] for r in rs):.1f} | {max(r['answer'] for r in rs)} |")
    L.append("")
    train = all_rows.get("train", [])
    if train:
        ops = Counter()
        for r in train:
            for sym, arg in zip(r["trace"][1::2], r["trace"][2::2]):
                ops[f"{sym}{arg}" if sym in "*/" else sym] += 1
        total = sum(ops.values())
        L.append("## Operaciones en train\n")
        L.append("| Operación | Frecuencia |")
        L.append("|---|---|")
        names = {"+": "suma", "-": "resta", "*2": "doble", "*3": "triple", "/2": "mitad", "/3": "tercera parte",
                 "/4": "cuarta parte"}
        for k, v in ops.most_common():
            L.append(f"| {names.get(k, k)} | {v / total:.1%} |")
        L.append("")
        answers = Counter(r["answer"] for r in train)
        L.append("## Respuestas en train\n")
        L.append(f"- Valores distintos: {len(answers)}; las 5 más frecuentes: "
                 + ", ".join(f"{a} ({c / len(train):.1%})" for a, c in answers.most_common(5)))
        L.append(f"- Respuestas de una cifra: {sum(c for a, c in answers.items() if a < 10) / len(train):.1%}; "
                 f"de dos cifras: {sum(c for a, c in answers.items() if 10 <= a < 100) / len(train):.1%}; "
                 f"de tres cifras: {sum(c for a, c in answers.items() if a >= 100) / len(train):.1%}")
        templates = Counter(r["template"].split("|")[0] for r in train)
        L.append(f"- Marcos: " + ", ".join(f"{k} {v / len(train):.0%}" for k, v in templates.items()))
        L.append("")
    L += _baselines_section(all_rows)
    L.append("## Ejemplos\n")
    for name, rows in all_rows.items():
        shown = set()
        for r in rows:
            key = (r["level"], r["type"])
            if key in shown:
                continue
            shown.add(key)
            L.append(f"- **{name} · nivel {r['level']} · tipo {r['type']}"
                     f"{' (desordenado)' if r['shuffled'] else ''}**: {r['question']} → **{r['answer']}**")
        L.append("")
    return "\n".join(L) + "\n"


def set_ranges(max_operand, max_val):
    """Cambia el rango de los números (para calibrar sin tocar el código)."""
    global MAX_OPERAND, MAX_VAL
    MAX_OPERAND, MAX_VAL = max_operand, max_val


def main():
    ap = argparse.ArgumentParser(description="Genera el dataset de problemas en castellano.")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n_train", type=int, default=50_000, help="problemas de train POR NIVEL")
    ap.add_argument("--n_val", type=int, default=1_000, help="por nivel")
    ap.add_argument("--n_test", type=int, default=2_000, help="por nivel, para cada fichero de test")
    ap.add_argument("--sample", action="store_true", help="muestra pequeña: 50 por nivel en cada fichero")
    ap.add_argument("--max_operand", type=int, default=MAX_OPERAND,
                    help=f"operando máximo de sumas y restas y valor inicial máximo (por defecto {MAX_OPERAND})")
    ap.add_argument("--max_val", type=int, default=MAX_VAL,
                    help=f"valor máximo de cualquier cantidad del problema (por defecto {MAX_VAL})")
    a = ap.parse_args()
    set_ranges(a.max_operand, a.max_val)
    if a.sample:
        a.n_train = a.n_val = a.n_test = 50
    generate_dataset(a.out, a.n_train, a.n_val, a.n_test, a.seed)


if __name__ == "__main__":
    main()
