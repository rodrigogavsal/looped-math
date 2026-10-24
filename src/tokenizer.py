"""Tokenizador por caracteres con vocabulario fijo según Sección 2.3 del contrato común.

IDs especiales:
  <pad> = 0
  <eos> = 1
  <unk> = 2
Seguido de unos 100 caracteres en orden fijo:
  espacio, salto de línea, 0-9, a-z, A-Z, á é í ó ú ü ñ Á É Í Ó Ú Ü Ñ, . ; : ¿ ? ¡ ! ( ) - " ' + × =
"""

from typing import List, Union


class CharTokenizer:
    SPECIAL_TOKENS = ["<pad>", "<eos>", "<unk>"]
    PAD_ID = 0
    EOS_ID = 1
    UNK_ID = 2

    def __init__(self):
        # 1. Especiales
        vocab = list(self.SPECIAL_TOKENS)

        # 2. Espacio y salto de línea
        vocab.extend([" ", "\n"])

        # 3. Dígitos 0-9
        vocab.extend([str(i) for i in range(10)])

        # 4. Letras minúsculas a-z
        vocab.extend([chr(c) for c in range(ord("a"), ord("z") + 1)])

        # 5. Letras mayúsculas A-Z
        vocab.extend([chr(c) for c in range(ord("A"), ord("Z") + 1)])

        # 6. Vocales con tilde y caracteres en español (minúsculas y mayúsculas)
        vocab.extend(["á", "é", "í", "ó", "ú", "ü", "ñ"])
        vocab.extend(["Á", "É", "Í", "Ó", "Ú", "Ü", "Ñ"])

        # 7. Signos de puntuación y matemáticos
        # Nota: '×' es el signo de multiplicación español/matemático
        punctuation = [",", ".", ";", ":", "¿", "?", "¡", "!", "(", ")", "-", '"', "'", "+", "×", "="]
        for p in punctuation:
            if p not in vocab:
                vocab.append(p)

        self.vocab = vocab
        self.char_to_id = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.id_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}

        self.vocab_size = len(self.vocab)
        self.pad_id = self.PAD_ID
        self.eos_id = self.EOS_ID
        self.unk_id = self.UNK_ID

    def encode(self, text: str) -> List[int]:
        """Convierte una cadena de texto en una lista de enteros (IDs de tokens)."""
        tokens = []
        for ch in text:
            # Tolerancia: si se usa '*' convertirlo a '×' para robustez aritmética
            if ch == "*":
                ch = "×"
            tokens.append(self.char_to_id.get(ch, self.unk_id))
        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = False) -> str:
        """Convierte una lista de enteros (IDs de tokens) en una cadena de texto."""
        chars = []
        for tok in token_ids:
            if isinstance(tok, int):
                tid = tok
            else:
                tid = int(tok)

            if tid == self.pad_id:
                if not skip_special:
                    chars.append("")
            elif tid == self.eos_id:
                if not skip_special:
                    chars.append("<eos>")
            elif tid == self.unk_id:
                if not skip_special:
                    chars.append("<unk>")
            else:
                chars.append(self.id_to_char.get(tid, ""))
        return "".join(chars)

    def __len__(self):
        return self.vocab_size


# Instancia por defecto
_default_tokenizer = None


def get_tokenizer() -> CharTokenizer:
    global _default_tokenizer
    if _default_tokenizer is None:
        _default_tokenizer = CharTokenizer()
    return _default_tokenizer
