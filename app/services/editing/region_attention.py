"""Region-aware cross-attention for DiT blocks.

The leakage this fixes: in a DiT, every text token attends to *every* image token,
so "change the shirt to red" pushes red into the background too. The fix is an
additive bias on the cross-attention logits that suppresses the edit-carrying text
tokens over image positions outside the edit region.

Training-free: it only reshapes attention logits, so it runs against an existing
checkpoint. It is applied to cross-attention (`attn2`) only - self-attention still
sees the whole frame, which is what keeps the edit globally coherent (lighting,
perspective) instead of looking pasted in.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from app.services.editing.masks import as_soft_mask, resize_mask

logger = logging.getLogger(__name__)

# Penalty applied in logit space. 12 nats is ~6e-6 relative weight - effectively a
# hard mask - while intermediate strengths stay meaningfully graded, which a
# -10_000 sentinel would not (it saturates the softmax at any strength > 0).
MAX_LOGIT_PENALTY = 12.0


def latent_grid(num_tokens: int, aspect: float = 1.0) -> tuple[int, int]:
    """Find the (height, width) factorisation of num_tokens closest to aspect."""
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    target_aspect = max(float(aspect), 1e-3)
    best_pair = (1, num_tokens)
    best_error = float("inf")

    for h in range(1, int(math.isqrt(num_tokens)) + 1):
        if num_tokens % h == 0:
            w = num_tokens // h
            for cand_h, cand_w in ((h, w), (w, h)):
                cand_aspect = cand_w / cand_h
                error = abs(math.log(cand_aspect / target_aspect))
                if error < best_error:
                    best_error = error
                    best_pair = (cand_h, cand_w)

    height, width = best_pair
    if height == 1 and num_tokens > 16:
        logger.warning(
            "latent_grid(%d) degenerated to a 1x%d strip; pass an explicit "
            "aspect or a factorable token count, or masks will be meaningless.",
            num_tokens,
            width,
        )
    return height, width


def mask_to_token_weights(
    mask: torch.Tensor,
    num_tokens: int,
    *,
    aspect: float = 1.0,
) -> torch.Tensor:
    """Flatten a spatial mask onto the DiT's token sequence -> (num_tokens,) in [0,1]."""
    height, width = latent_grid(num_tokens, aspect)
    resized = resize_mask(as_soft_mask(mask), height, width)
    return resized.reshape(-1)[:num_tokens]


def region_attention_bias(
    mask: torch.Tensor,
    *,
    num_image_tokens: int,
    num_text_tokens: int,
    edit_token_indices: torch.Tensor | list[int] | None = None,
    strength: float = 1.0,
    aspect: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Additive cross-attention bias of shape (num_image_tokens, num_text_tokens).

    Entries are <= 0. Outside the mask, the selected edit tokens are pushed down by
    up to `strength` * MAX_LOGIT_PENALTY nats, so those words stop steering that
    region. Tokens not selected (global style, subject identity) are never
    suppressed. `strength` interpolates: 0 disables, 1 is effectively a hard mask.
    """
    if strength <= 0:
        return torch.zeros(num_image_tokens, num_text_tokens, dtype=dtype)
    weights = mask_to_token_weights(mask, num_image_tokens, aspect=aspect).to(dtype)
    outside = (1.0 - weights).clamp(0.0, 1.0)  # 1 where the source must be preserved

    bias = torch.zeros(num_image_tokens, num_text_tokens, dtype=dtype)
    if edit_token_indices is None:
        columns = torch.arange(num_text_tokens)
    else:
        columns = torch.as_tensor(list(edit_token_indices), dtype=torch.long)
        columns = columns[(columns >= 0) & (columns < num_text_tokens)]
    if columns.numel() == 0:
        return bias
    penalty = -MAX_LOGIT_PENALTY * float(strength)
    bias[:, columns] = (outside * penalty).unsqueeze(1)
    return bias


@dataclass
class AttentionCapture:
    """Accumulates cross-attention probabilities so a mask can be inferred from them."""

    num_image_tokens: int | None = None
    num_text_tokens: int | None = None
    _total: torch.Tensor | None = field(default=None, repr=False)
    _count: int = 0

    def add(self, probabilities: torch.Tensor) -> None:
        """Add one layer's attention. Accepts (..., image_tokens, text_tokens)."""
        maps = probabilities.detach().float()
        maps = maps.reshape(-1, maps.shape[-2], maps.shape[-1]).mean(dim=0)
        if self._total is None:
            self._total = torch.zeros_like(maps)
            self.num_image_tokens, self.num_text_tokens = maps.shape
        elif maps.shape != self._total.shape:
            return  # different resolution block; ignore rather than misalign
        self._total += maps
        self._count += 1

    @property
    def mean_map(self) -> torch.Tensor | None:
        if self._total is None or self._count == 0:
            return None
        return self._total / self._count

    def reset(self) -> None:
        self._total = None
        self._count = 0

    def token_mask(
        self,
        token_indices: list[int] | torch.Tensor,
        *,
        aspect: float = 1.0,
        percentile: float = 0.7,
    ) -> torch.Tensor | None:
        """Turn the captured attention for `token_indices` into a soft (1,1,h,w) mask.

        This is the Prompt-to-Prompt style inference used when the caller gives no
        explicit mask: whatever the edit words already attend to *is* the region.
        """
        maps = self.mean_map
        if maps is None:
            return None
        columns = torch.as_tensor(list(token_indices), dtype=torch.long)
        columns = columns[(columns >= 0) & (columns < maps.shape[1])]
        if columns.numel() == 0:
            return None
        pooled = maps[:, columns].mean(dim=1)
        span = pooled.max() - pooled.min()
        pooled = (pooled - pooled.min()) / span if float(span) > 0 else torch.zeros_like(pooled)
        # Keep the strongest responses; attention is diffuse and a raw map is noisy.
        threshold = torch.quantile(pooled, min(max(percentile, 0.0), 1.0))
        pooled = ((pooled - threshold) / (1.0 - threshold + 1e-6)).clamp(0.0, 1.0)
        height, width = latent_grid(pooled.numel(), aspect)
        return pooled.reshape(1, 1, height, width)


class RegionAwareAttnProcessor:
    """Wraps a diffusers attention processor and injects the region bias.

    Attach with `transformer.set_attn_processor({name: RegionAwareAttnProcessor(...)})`
    for cross-attention layers only. `set_region` is called per denoise step so the
    same processor can follow a schedule.
    """

    def __init__(
        self,
        base_processor,
        *,
        capture: AttentionCapture | None = None,
    ):
        self.base_processor = base_processor
        self.capture = capture
        self._bias: torch.Tensor | None = None
        self._warned_shape = False

    def set_bias(self, bias: torch.Tensor | None) -> None:
        self._bias = bias
        self._warned_shape = False

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        if encoder_hidden_states is not None and self._bias is not None:
            bias = self._bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
            expected = (hidden_states.shape[1], encoder_hidden_states.shape[1])
            batch_size = hidden_states.shape[0]
            if tuple(bias.shape[-2:]) == expected:
                # diffusers' prepare_attention_mask repeat-interleaves by head count
                # and then views as (batch, heads, q, k), so a (1, q, k) bias has too
                # few elements for batch > 1 and raises. Classifier-free guidance
                # batches uncond+cond into one forward, making batch=2 the norm, so
                # expand to the real batch here. A caller that wants a different bias
                # per batch row (e.g. no masking on the unconditional half) can set a
                # bias that already carries the batch dimension.
                if bias.ndim == 2:
                    bias = bias.unsqueeze(0)
                if bias.shape[0] == 1 and batch_size > 1:
                    bias = bias.expand(batch_size, -1, -1)
                attention_mask = bias if attention_mask is None else attention_mask + bias
            elif not self._warned_shape:
                # Dropping the bias here disables region masking for the whole run,
                # which looks exactly like "the feature does not work". Say so once.
                self._warned_shape = True
                logger.warning(
                    "Region bias %s does not match this block's (image=%d, text=%d); "
                    "region masking is INACTIVE. Build the bias with the transformer's "
                    "own token count, not the mask resolution.",
                    tuple(bias.shape[-2:]),
                    expected[0],
                    expected[1],
                )
        return self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Role-aware masking (Prompt-to-Prompt style)
# ---------------------------------------------------------------------------

TokenRole = str  # "edit_target" | "context" | "neutral"

_EDIT_VERBS = re.compile(
    r"\b(change|make|turn|recolor|replace|convert|set|swap|repaint|restyle)\b", re.IGNORECASE
)
_CONTEXT_CUES = re.compile(
    r"\b(keep|preserve|maintain|retain|leave|without|while|but|except|unchanged)\b",
    re.IGNORECASE,
)
_ROLE_STOPWORDS = frozenset(
    """a an the this that these those and or of to in on at by for with from into it its
    is are be was were do does did please can you my our your their his her same as""".split()
)


def classify_token_roles(prompt: str) -> tuple[tuple[str, TokenRole], ...]:
    """Heuristic word-level role tagging: (word, role) pairs in prompt order.

    Words after an edit verb are what the user wants changed; words after a
    preservation cue ("keep the background neutral") describe what must survive and
    are therefore context, not targets. A learned or POS-based classifier can
    replace this without touching the bias builder - the interface is the roles.
    """
    words = re.findall(r"[a-zA-Z0-9]+", prompt)
    roles: list[tuple[str, TokenRole]] = []
    in_context_clause = False
    seen_edit_verb = False
    for word in words:
        lowered = word.lower()
        if _CONTEXT_CUES.fullmatch(lowered):
            in_context_clause = True
            roles.append((word, "neutral"))
            continue
        if _EDIT_VERBS.fullmatch(lowered):
            seen_edit_verb = True
            in_context_clause = False
            roles.append((word, "neutral"))
            continue
        if lowered in _ROLE_STOPWORDS or len(lowered) < 3:
            roles.append((word, "neutral"))
            continue
        if in_context_clause:
            roles.append((word, "context"))
        elif seen_edit_verb or not roles:
            roles.append((word, "edit_target"))
        else:
            roles.append((word, "edit_target"))
    return tuple(roles)


def extract_edit_mask(
    cross_attn_maps: torch.Tensor,
    edit_keywords: Sequence[int],
    *,
    threshold: float = 0.35,
    aspect: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive the edit region from captured cross-attention.

    `cross_attn_maps` is (..., image_tokens, text_tokens). Returns
    `(soft_mask, binary_mask)`, both (1,1,h,w) - the soft one drives blending and
    guidance, the binary one is for IoU reporting and hard masking.
    """
    maps = cross_attn_maps.detach().float()
    maps = maps.reshape(-1, maps.shape[-2], maps.shape[-1]).mean(dim=0)
    columns = torch.as_tensor(list(edit_keywords), dtype=torch.long)
    columns = columns[(columns >= 0) & (columns < maps.shape[1])]
    if columns.numel() == 0:
        empty = torch.zeros(1, 1, *latent_grid(maps.shape[0], aspect))
        return empty, empty.clone()
    activation = maps[:, columns].mean(dim=1)
    span = activation.max() - activation.min()
    if float(span) > 0:
        activation = (activation - activation.min()) / span
    else:
        activation = torch.zeros_like(activation)
    height, width = latent_grid(activation.numel(), aspect)
    soft = activation.reshape(1, 1, height, width)
    return soft, (soft > threshold).float()


def build_attention_bias(
    edit_mask: torch.Tensor,
    token_roles: Sequence[TokenRole],
    *,
    leak_penalty: float = -12.0,
    context_boost: float = 0.5,
    num_image_tokens: int | None = None,
    aspect: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Role-aware additive bias, shaped (num_text_tokens, num_image_tokens).

    * `edit_target` tokens are penalised **outside** the region - they stop
      steering pixels they were never about.
    * `context` tokens are boosted **inside** the region, so the new content stays
      consistent with the context the user described rather than ignoring it.
    * `neutral` tokens are left alone.

    `leak_penalty` defaults to -12 nats (~6e-6 relative weight) rather than the
    -1e4 of the reference sketch: -1e4 saturates the softmax so the knob cannot be
    tuned, and in fp16 it overflows to -inf, which produces NaNs when an entire row
    is masked.
    """
    if num_image_tokens is None:
        # The mask resolution is only coincidentally the transformer's token count
        # (PixArt-512 patches a 64x64 latent down to 1024 tokens, not 4096). Callers
        # should pass it explicitly; warn rather than produce a bias that will be
        # silently dropped downstream.
        num_image_tokens = int(as_soft_mask(edit_mask).numel())
        logger.debug(
            "build_attention_bias defaulted num_image_tokens to the mask size (%d); "
            "pass the transformer's token count explicitly.",
            num_image_tokens,
        )
    weights = mask_to_token_weights(edit_mask, num_image_tokens, aspect=aspect).to(dtype)
    outside = (1.0 - weights).clamp(0.0, 1.0)

    bias = torch.zeros(len(token_roles), num_image_tokens, dtype=dtype)
    for index, role in enumerate(token_roles):
        if role == "edit_target":
            bias[index] = outside * float(leak_penalty)
        elif role == "context":
            bias[index] = weights * float(context_boost)
    return bias


def masked_cross_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_bias: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Cross-attention with an additive region bias.

    `query` is (..., image_tokens, dim), `key`/`value` are (..., text_tokens, dim),
    and `attention_bias` is (text_tokens, image_tokens) as built above - it is
    transposed here so callers can pass the builder's output unchanged.
    """
    scale = scale if scale is not None else query.shape[-1] ** -0.5
    logits = (query @ key.transpose(-1, -2)) * scale
    if attention_bias is not None:
        bias = attention_bias.transpose(-1, -2).to(device=logits.device, dtype=logits.dtype)
        logits = logits + bias
    return logits.softmax(dim=-1) @ value
