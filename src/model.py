"""Arquitectura GPT estándar y Looped Transformer con instrumentación interna."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


DEFAULT_MODEL_CFG: Dict[str, Any] = {
    "arch": "looped",          # "gpt" | "looped"
    "vocab_size": 97,          # Tamaño del vocabulario fijo de caracteres (Sección 2.3)
    "context_length": 512,     # Longitud máxima de secuencia
    "emb_dim": 256,            # Dimensión del estado oculto (d)
    "n_heads": 8,              # Número de cabezas de atención (head_dim = 32)
    "n_layers": 1,             # Bloques con pesos distintos (L en GPT, k=1 en Looped)
    "n_unique_layers": 1,      # Alias de n_layers para explicitar k bloques únicos en Looped
    "n_loops": 4,              # Iteraciones del bucle recurrente (N en Looped)
    "input_injection": False,  # Suma opcional del embedding inicial h_0 en cada iteración t > 0
    "drop_rate": 0.0,          # Sin dropout sobre datos sintéticos
    "qkv_bias": False,         # Sin sesgo en las proyecciones Q, K, V
}


class MultiHeadAttention(nn.Module):
    """Atención multi-cabeza causal con extracción opcional de mapas de atención.

    Dos caminos que calculan lo mismo:
      - return_attn=False (entrenamiento y evaluación normal): F.scaled_dot_product_attention.
        No materializa la matriz de pesos (b, h, T, T), así que gasta mucha menos memoria;
        con la versión explícita, GPT de 4 capas y batch 64 no cabe en una GPU de 4 GB.
      - return_attn=True (análisis): la versión explícita del notebook, que devuelve los pesos.
    La diferencia numérica entre ambos caminos es de ~1e-6 (lo comprueba tests/test_model.py).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("d_out debe ser divisible por num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, num_tokens, _ = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        if not return_attn:
            # Camino eficiente en memoria (mismo cálculo que el explícito de abajo).
            context_vec = F.scaled_dot_product_attention(
                queries, keys, values,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
            context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
            return self.out_proj(context_vec)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        context_vec = (attn_weights_dropped @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        if return_attn:
            return context_vec, attn_weights
        return context_vec


class LayerNorm(nn.Module):
    """Normalización por capa (Layer Normalization)."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """Activación Gaussian Error Linear Unit (aproximación con tanh)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Misma fórmula que el notebook (0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))),
        # pero con el kernel fusionado de PyTorch: no guarda los intermedios y ahorra memoria.
        return F.gelu(x, approximate="tanh")


class FeedForward(nn.Module):
    """Red feed-forward posicional con expansión 4x."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Bloque Transformer Pre-LayerNorm con conexiones residuales."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        shortcut = x
        x = self.norm1(x)
        if return_attn:
            x, attn_weights = self.att(x, return_attn=True)
        else:
            x = self.att(x, return_attn=False)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        if return_attn:
            return x, attn_weights
        return x


class GPTModel(nn.Module):
    """Modelo GPT estándar de L capas con pesos independientes por capa."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # Checkpointing de activaciones: si es True, al entrenar no se guardan las activaciones
        # internas de cada bloque y se recalculan en el backward (menos memoria, ~30 % más lento).
        # No cambia el resultado. Lo activa train.py con cfg["grad_checkpoint"].
        self.grad_checkpoint = bool(cfg.get("grad_checkpoint", False))

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False)
        return block(x, return_attn=False)

    def embed(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Calcula el estado inicial h_0 sumando embeddings de token y posición."""
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        return self.drop_emb(tok_embeds + pos_embeds)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """Proyecta cualquier estado oculto intermedio o final a logits del vocabulario (Logit Lens)."""
        return self.out_head(self.final_norm(h))

    def forward(
        self,
        in_idx: torch.Tensor,
        n_loops: Optional[int] = None,
        return_hidden: bool = False,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]]:
        x = self.embed(in_idx)
        if not return_hidden and not return_attn:
            for block in self.trf_blocks:
                x = self._run_block(block, x)
            return self.head(x)

        extras: Dict[str, List[torch.Tensor]] = {}
        if return_hidden:
            extras["hidden"] = [x]
        if return_attn:
            extras["attn"] = []

        for block in self.trf_blocks:
            if return_attn:
                x, attn_w = block(x, return_attn=True)
                extras["attn"].append(attn_w)
            else:
                x = block(x, return_attn=False)
            if return_hidden:
                extras["hidden"].append(x)

        return self.head(x), extras


class LoopedGPT(GPTModel):
    """
    Looped Transformer que hereda de GPTModel y aplica sus k bloques de forma
    recurrente durante N iteraciones (con pesos compartidos entre vueltas).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        full_cfg = {**DEFAULT_MODEL_CFG, **cfg}
        if "n_unique_layers" in cfg:
            k_blocks = int(cfg["n_unique_layers"])
        elif "n_layers" in cfg:
            k_blocks = int(cfg["n_layers"])
        else:
            k_blocks = 1

        self.n_loops = int(full_cfg["n_loops"])
        self.n_unique_layers = k_blocks
        self.n_layers = k_blocks
        self.input_injection = bool(full_cfg.get("input_injection", False))

        full_cfg["n_layers"] = k_blocks
        full_cfg["n_unique_layers"] = k_blocks
        super().__init__(full_cfg)
        self.cfg = copy.deepcopy(full_cfg)

    @property
    def shared_blocks(self) -> nn.Sequential:
        """Alias hacia self.trf_blocks para explicitar que los pesos se comparten sin duplicarlos en state_dict."""
        return self.trf_blocks

    def step(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        t: int,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Aplica una iteración completa de los k bloques compartidos."""
        if self.input_injection and t > 0:
            h = h + e
        attns: Optional[List[torch.Tensor]] = [] if return_attn else None
        for block in self.shared_blocks:
            if return_attn and attns is not None:
                h, attn_w = block(h, return_attn=True)
                attns.append(attn_w)
            else:
                h = self._run_block(block, h)
        return (h, attns) if (return_attn and attns is not None) else h

    def forward(
        self,
        in_idx: torch.Tensor,
        n_loops: Optional[int] = None,
        return_hidden: bool = False,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]]:
        loops = self.n_loops if n_loops is None else int(n_loops)
        e = self.embed(in_idx)
        h = e

        if not return_hidden and not return_attn:
            for t in range(loops):
                h = self.step(h, e, t, return_attn=False)  # type: ignore[assignment]
            return self.head(h)

        extras: Dict[str, List[torch.Tensor]] = {}
        if return_hidden:
            extras["hidden"] = [h]
        if return_attn:
            extras["attn"] = []

        for t in range(loops):
            if return_attn:
                h, step_attns = self.step(h, e, t, return_attn=True)  # type: ignore[misc]
                extras["attn"].extend(step_attns)
            else:
                h = self.step(h, e, t, return_attn=False)  # type: ignore[assignment]
            if return_hidden:
                extras["hidden"].append(h)

        return self.head(h), extras


def _infer_arch(cfg: Dict[str, Any]) -> str:
    """Infiere la arquitectura ('gpt' o 'looped') a partir de las claves presentes en cfg."""
    if "arch" in cfg:
        return str(cfg["arch"])
    if "n_loops" in cfg and "n_layers" not in cfg:
        return "looped"
    return "gpt"


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Factoría unificada que construye GPTModel o LoopedGPT a partir de un diccionario cfg."""
    arch = _infer_arch(cfg)
    full_cfg = {**DEFAULT_MODEL_CFG, **cfg, "arch": arch}
    if arch == "gpt":
        return GPTModel(full_cfg)
    if arch == "looped":
        return LoopedGPT(full_cfg)
    raise ValueError(f"Arquitectura desconocida: {arch!r}. Usa 'gpt' o 'looped'.")


def count_params(
    model: nn.Module,
    include_embeddings: bool = True,
    trainable_only: bool = True,
) -> int:
    """Cuenta el número total de parámetros del modelo (por defecto, los entrenables incluyendo embeddings)."""
    params = (p for p in model.parameters() if (p.requires_grad or not trainable_only))
    total = sum(p.numel() for p in params)
    if not include_embeddings and hasattr(model, "tok_emb") and hasattr(model, "pos_emb"):
        total -= model.tok_emb.weight.numel() + model.pos_emb.weight.numel()
    return total


def effective_depth(cfg: Dict[str, Any]) -> int:
    """Profundidad computacional efectiva: L en GPT estándar, k * N en Looped."""
    arch = _infer_arch(cfg)
    if arch == "gpt":
        return int(cfg["n_layers"])
    k = int(cfg.get("n_unique_layers", cfg.get("n_layers", 1)))
    return k * int(cfg["n_loops"])


def block_flops_per_token(cfg: Dict[str, Any], seq_len: Optional[int] = None) -> int:
    """
    FLOPs aproximados (2 * MACs) por token en un único bloque Transformer:
      - Proyecciones Q, K, V y salida: 4 * (2 * d^2) = 8 * d^2
      - Atención QK^T y AV:            2 * (2 * T * d) = 4 * T * d
      - Red FeedForward (d -> 4d -> d): 2 * (2 * 4 * d^2) = 16 * d^2
      Total por bloque = 24 * d^2 + 4 * T * d
    """
    d = int(cfg["emb_dim"])
    T = int(cfg.get("context_length", 512) if seq_len is None else seq_len)
    return 24 * (d ** 2) + 4 * T * d


def gpt_flops_per_token(cfg: Dict[str, Any], seq_len: Optional[int] = None) -> int:
    """FLOPs totales de forward por token para un GPTModel de L capas."""
    L = int(cfg["n_layers"])
    d = int(cfg["emb_dim"])
    V = int(cfg.get("vocab_size", DEFAULT_MODEL_CFG["vocab_size"]))
    return L * block_flops_per_token(cfg, seq_len=seq_len) + 2 * d * V


def flops_per_token(
    cfg: Dict[str, Any],
    seq_len: Optional[int] = None,
    depth: Optional[int] = None,
) -> int:
    """FLOPs totales de forward por token incluyendo todos los pasos efectivos y la cabeza."""
    full_cfg = {**DEFAULT_MODEL_CFG, **cfg, "arch": _infer_arch(cfg)}
    d = int(full_cfg["emb_dim"])
    V = int(full_cfg["vocab_size"])
    steps = effective_depth(full_cfg) if depth is None else int(depth)
    return steps * block_flops_per_token(full_cfg, seq_len=seq_len) + 2 * d * V


model_flops_per_token = flops_per_token


def gpt_cfg_from(cfg: Dict[str, Any], n_layers: int) -> Dict[str, Any]:
    """Crea una configuración de GPTModel estándar con n_layers capas."""
    full_cfg = {**DEFAULT_MODEL_CFG, **cfg}
    keys = ("vocab_size", "context_length", "emb_dim", "n_heads", "drop_rate", "qkv_bias")
    out = {k: full_cfg[k] for k in keys}
    out["arch"] = "gpt"
    out["n_layers"] = int(n_layers)
    out["n_loops"] = 1
    return out


@torch.no_grad()
def unroll_to_gpt(
    looped_model: LoopedGPT,
    cfg: Optional[Dict[str, Any]] = None,
    n_loops: Optional[int] = None,
) -> GPTModel:
    """
    Desenrolla un LoopedGPT en un GPTModel estándar de profundidad efectiva k * N
    copiando los pesos de los bloques compartidos en cada capa.
    """
    if looped_model.input_injection:
        raise ValueError("unroll_to_gpt requiere input_injection=False")
    base_cfg = looped_model.cfg if cfg is None else cfg
    loops = looped_model.n_loops if n_loops is None else int(n_loops)
    k = looped_model.n_unique_layers
    total_layers = k * loops
    gpt = GPTModel(gpt_cfg_from(base_cfg, n_layers=total_layers))

    gpt.tok_emb.load_state_dict(looped_model.tok_emb.state_dict())
    gpt.pos_emb.load_state_dict(looped_model.pos_emb.state_dict())
    gpt.final_norm.load_state_dict(looped_model.final_norm.state_dict())
    gpt.out_head.load_state_dict(looped_model.out_head.state_dict())

    for i in range(total_layers):
        gpt.trf_blocks[i].load_state_dict(
            looped_model.shared_blocks[i % k].state_dict()
        )
    return gpt


@torch.no_grad()
def looped_from_gpt(
    gpt_model: GPTModel,
    cfg: Optional[Dict[str, Any]] = None,
    n_loops: Optional[int] = None,
    init: str = "first",
    mode: Optional[str] = None,
) -> LoopedGPT:
    """
    Construye un LoopedGPT a partir de un GPTModel inicializando los bloques
    compartidos mediante 'first', 'last', 'middle', 'mean' o 'stride'.
    """
    chosen_mode = mode if mode is not None else init
    L = len(gpt_model.trf_blocks)
    base_cfg = copy.deepcopy(gpt_model.cfg if cfg is None else cfg)
    base_cfg["arch"] = "looped"
    base_cfg.setdefault("n_unique_layers", 1)
    base_cfg["n_layers"] = base_cfg["n_unique_layers"]
    if n_loops is not None:
        base_cfg["n_loops"] = int(n_loops)
    else:
        base_cfg.setdefault("n_loops", L)

    looped = LoopedGPT(base_cfg)
    looped.tok_emb.load_state_dict(gpt_model.tok_emb.state_dict())
    looped.pos_emb.load_state_dict(gpt_model.pos_emb.state_dict())
    looped.final_norm.load_state_dict(gpt_model.final_norm.state_dict())
    looped.out_head.load_state_dict(gpt_model.out_head.state_dict())

    k = looped.n_unique_layers
    if chosen_mode == "first":
        for j in range(k):
            looped.shared_blocks[j].load_state_dict(gpt_model.trf_blocks[j].state_dict())
    elif chosen_mode == "last":
        for j in range(k):
            looped.shared_blocks[j].load_state_dict(gpt_model.trf_blocks[L - k + j].state_dict())
    elif chosen_mode == "middle":
        start = max(0, (L - k) // 2)
        for j in range(k):
            looped.shared_blocks[j].load_state_dict(gpt_model.trf_blocks[start + j].state_dict())
    elif chosen_mode == "stride":
        stride = max(1, L // k)
        for j in range(k):
            looped.shared_blocks[j].load_state_dict(gpt_model.trf_blocks[min(j * stride, L - 1)].state_dict())
    elif chosen_mode == "mean":
        for j in range(k):
            idxs = list(range(j, L, k))
            avg_sd = {}
            for key in gpt_model.trf_blocks[0].state_dict():
                avg_sd[key] = sum(gpt_model.trf_blocks[i].state_dict()[key].float() for i in idxs) / len(idxs)
                avg_sd[key] = avg_sd[key].to(gpt_model.trf_blocks[0].state_dict()[key].dtype)
            looped.shared_blocks[j].load_state_dict(avg_sd)
    else:
        raise ValueError(f"Modo de inicialización desconocido: {chosen_mode!r}")

    return looped
