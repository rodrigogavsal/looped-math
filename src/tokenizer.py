"""Tokenizador por caracteres con vocabulario FIJO (contrato común, sección 2.3).

El vocabulario no depende de los datos: así las partes 2, 3 y 4 pueden trabajar
antes de que exista el dataset. El generador de problemas solo usa estos
caracteres (lo comprueban los tests de datos).

IDs: <pad> = 0, <eos> = 1, <unk> = 2, y después los caracteres en el orden de CHARS.

Formato de un ejemplo (contrato, sección 2.2):
    "Pregunta: " + pregunta + "\nRespuesta: " + str(respuesta)   seguido de <eos>
La pérdida solo se calcula sobre los caracteres de la respuesta y el <eos>.
"""

SPECIALS = ["<pad>", "<eos>", "<unk>"]
CHARS = list(
    " \n"
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "áéíóúüñÁÉÍÓÚÜÑ"
    ".,;:¿?¡!()-\"'+×="
)

PROMPT_PREFIX = "Pregunta: "
ANSWER_PREFIX = "\nRespuesta: "
MAX_LEN = 512  # tokens por ejemplo, incluido el <eos> (= context_length del modelo)


def format_prompt(question):
    """Lo que recibe el modelo en evaluación: todo hasta 'Respuesta: ' incluido."""
    return PROMPT_PREFIX + question + ANSWER_PREFIX


def format_example(question, answer):
    """Texto completo de entrenamiento, SIN el <eos> (se añade al tokenizar)."""
    return format_prompt(question) + str(answer)


def example_length(question, answer):
    """Nº de tokens del ejemplo completo, incluido el <eos>."""
    return len(format_example(question, answer)) + 1


class CharTokenizer:
    """Tokenizador por caracteres. Único para todo el repositorio (Partes 1-4)."""

    SPECIAL_TOKENS = SPECIALS
    PAD_ID, EOS_ID, UNK_ID = 0, 1, 2

    def __init__(self):
        self.itos = SPECIALS + CHARS
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad_id, self.eos_id, self.unk_id = self.PAD_ID, self.EOS_ID, self.UNK_ID
        self.vocab_size = len(self.itos)
        # Alias con los nombres que usa el código de las Partes 2 y 3
        self.vocab = self.itos
        self.char_to_id = self.stoi
        self.id_to_char = dict(enumerate(self.itos))

    def __len__(self):
        return self.vocab_size

    def encode(self, text, add_eos=False):
        # '*' se normaliza a '×' (el dataset no usa ninguno de los dos; es solo por robustez)
        ids = [self.stoi.get("×" if c == "*" else c, self.unk_id) for c in text]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids, skip_special=True):
        out = []
        for i in ids:
            i = int(i)
            if i < len(SPECIALS):
                if skip_special:
                    continue
                out.append(self.itos[i])
            else:
                out.append(self.itos[i])
        return "".join(out)

    def is_supported(self, text):
        """True si todos los caracteres están en el vocabulario (ninguno sería <unk>)."""
        return all(c in self.stoi and self.stoi[c] >= len(SPECIALS) for c in text)

    def unsupported_chars(self, text):
        return sorted({c for c in text if not (c in self.stoi and self.stoi[c] >= len(SPECIALS))})


_DEFAULT = None


def get_tokenizer():
    """Instancia compartida del tokenizador (la usan data.py, train.py, evaluate.py, run.py)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = CharTokenizer()
    return _DEFAULT
