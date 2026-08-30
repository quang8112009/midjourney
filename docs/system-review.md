# System review

Post-build review of the editing modules (`app/services/editing/`) and the chat
reasoning pipeline (`app/services/reasoning_service.py`). Every issue below was
**reproduced before being fixed** - none are speculative.

## 1. System state

| Area | State |
|---|---|
| Editing modules | 7 modules, 69 tests, 86% coverage; not yet run against a real checkpoint |
| Chat pipeline | two-pass, 84 tests, flag-switchable to single-pass |
| Gates | ruff clean, 158 tests, coverage 86% (gate 80), eval + 3 scripts exit 0 |
| Biggest gap | attention processor is not attached to a real transformer yet |

Data flow: `prompt + source (+mask, scene facts)` → `plan_edit` (one cheap pass:
region, scope, coefficients, alignment) → gate → denoise loop (attention bias →
spatial CFG → scheduled blend) → latents.

## 2. Issues found

### CRITICAL - ordinary "add" prompts were blocked as contradictions

*`alignment.py:108`.* The additive conflict check treated the indefinite article as a
count, so `"add a person"` parsed as "person number 1" and was blocked against a
3-person photo. Reproduced:

```
add another person to the photo  -> BLOCKED: asks to add person number 1, image has 3
add a person on the left         -> BLOCKED: asks to add person number 1, image has 3
```

**Impact:** a large class of legitimate edits refused outright, with an incoherent
message. This is the worst kind of failure here - the pre-generation gate exists to
save wasted work and instead denied valid work.

**Fix (applied):** split the vocabulary. Only **ordinals** ("second", "third") can
contradict an additive request; cardinals still govern removal ("remove three cats"
with one cat is still caught). Articles carry no count.

### HIGH - a prompt keyword silently disabled region masking

*`edit_planner.py:125`.* `classify_scope` promoted any prompt matching `_GLOBAL_HINTS`
("style", "all", …) to global scope, which sets `attention_strength = 0`. With an
explicit 1.5% user mask:

```
"change the jacket to a watercolor style" + user mask -> scope=global, masking OFF
```

**Impact:** the user drew a region, and one word turned off the entire leakage
prevention - producing exactly the failure the system was built to stop, silently.

**Fix (applied):** the keyword override is suppressed when the mask came from the
user. Coverage-based promotion (≥60% of frame) still applies in both cases.

### HIGH - a shape mismatch disabled masking with no signal

*`region_attention.py:205`, `:335`.* `build_attention_bias` defaults
`num_image_tokens` to the *mask resolution*, but a DiT's token count is different
(PixArt-512 patches a 64×64 latent to **1024** tokens, not 4096). The processor's
shape guard then dropped the bias - for every block, every step, silently.

**Impact:** region masking appears to run and does nothing. Indistinguishable from
"the feature doesn't work".

**Fix (applied):** the processor logs a one-shot warning naming both shapes and
stating masking is INACTIVE; the default logs at debug level. *Not* auto-corrected -
guessing the token count would hide a real integration error.

### MEDIUM - empty mask burned a full denoise loop

*`edit_planner.py:175`.* An all-zero user mask produced `scope=local`,
`attention_strength=1.0`, and a loop masked out everywhere - a degenerate result at
full cost. The `empty_mask` note was computed and never consumed.

**Fix (applied):** an empty mask is treated as "no region evidence" and takes the
existing `global_fallback` path, with a warning.

### MEDIUM - substring token matching selected the wrong words

*`edit_planner.py:115`.* Matching was bidirectional (`cleaned in term or term in
cleaned`), so the term `"red"` selected the token `"reduce"` and `"car"` selected
`"carpet"` - those tokens then got region-masked, steering the wrong words.

**Fix (applied):** exact match, plus sub-word pieces of ≥3 chars that *prefix* a term.

### MEDIUM - degenerate latent grid

*`region_attention.py:46`.* A non-factorable token count silently became a 1×N strip,
destroying the geometry every mask operation depends on. **Fix (applied):** warns.

### LOW - hardcoded values that belong in config

Not changed; see §4. `alignment.py:134-135` (`clarify_below=0.05`,
`assume_below=0.25`), `edit_planner.py:228` (the scope→attention-strength map),
`feather_radius`/`dilate_radius`, `changed_region_mask` threshold 0.05.

## 3. What was fixed

Six fixes, all small, backward compatible, and covered by 11 new regression tests
(`ReviewRegressionTests`). `classify_scope` gained a keyword-only argument defaulting
to today's behaviour, so existing callers are unaffected.

Verification after the fixes: **158 tests pass**, ruff clean, coverage 86%, and the
eval is unchanged (leakage 0.561 → 0.006), confirming the fixes did not weaken the
mechanism.

## 4. Follow-up work

### Done - items 1 and 3 (approved and implemented)

**1. Batched classifier-free guidance.** `edit_pipeline.py` now accepts
`denoise_pair=` (one forward returning uncond+cond) alongside the original
`denoise=`, via `batched_cfg_denoiser`. Backward compatible - existing callers are
untouched, and `_resolve_pair` adapts the old convention.

| | forwards (20 steps) | wall time | output |
|---|---|---|---|
| two-call | 40 | 93 ms | - |
| batched | 20 | 51 ms | bit-identical |

**3. Real-diffusers validation.** Attaching `RegionAwareAttnProcessor` to a real
`diffusers.models.attention_processor.Attention` exposed a hard failure: the
processor emitted a `(1, q, k)` bias, and `prepare_attention_mask`
repeat-interleaves by head count then views as `(batch, heads, q, k)`, so batch > 1
raised `RuntimeError`. Batch=1 passed, which is why unit tests never caught it - and
CFG batching makes batch=2 the normal case, so items 1 and 3 were blocking each
other. Fixed by expanding the bias to the real batch size; a caller may still supply
a bias that already carries a batch dimension. Verified at batch 1/2/4/8 against
real diffusers, with the bias confirmed to still change the output.

### Still needs confirmation
1. **Move thresholds into config.** An `EditingConfig` (pydantic, alongside
   `app/core/config.py`) for the §2-LOW values. **Trade-off:** more surface to
   document and validate; these are currently function defaults that are easy to
   override per call.
2. **Run against a real PixArt checkpoint.** The processor contract is now verified
   against real diffusers attention modules, but not against
   `PixArt-XL-2-512x512` weights on a GPU - block wiring, token counts, and the
   tuned parameters remain unvalidated end to end.
3. **Cache the SSIM gaussian window** (`metrics.py`) - rebuilt per call. Trivial win,
   only matters if metrics run in a hot loop.

## 5. Later review: the sub-word role alignment

`align_token_roles` (added after the first review) resolved sub-word pieces by
bidirectional substring matching, which reintroduced the defect fixed in
`locate_edit_tokens`: `"red"` claimed the token `"reduce"` and `"car"` claimed
`"carpet"`. That mattered more here, because roles feed `build_attention_bias`
directly - both false targets received the full -12 penalty outside the region,
steering on words the user never named.

String similarity is the wrong tool: a continuation piece like `"realistic"` is a
*suffix* of its parent word, so prefix rules miss it while looser rules bind
unrelated words. Replaced with `map_pieces_to_words`, which segments pieces into
words using the tokenizer's own scheme (WordPiece `##`, SentencePiece `\u2581`, BPE
`</w>`); each piece then inherits its own word's role, making cross-word
contamination structurally impossible. `locate_edit_tokens` shares the same
segmentation, so the two can no longer disagree.

## 6. Recommended next steps

Run against a real PixArt checkpoint on a CUDA worker. That is what turns every
number in `visual-reasoning-editing.md` from mechanism-level into product-level
evidence, and it is now the only thing blocking the remaining items - the parameters
should be tuned there before being frozen into config.
