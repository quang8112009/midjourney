# Prompt understanding

A structured stage that runs **before** any denoising. It turns a free-text edit
request into a JSON object the pipeline consumes directly - nothing downstream
re-reads the raw prompt.

Implementation: `app/services/editing/prompt_intent.py`. Tests:
`tests/test_prompt_intent.py`.

## 1. What was breaking

The planner treated a prompt as one bag of words. Three concrete failures, all
reproduced before this stage was written:

| Prompt | Old behaviour |
|---|---|
| `"change the shirt color to red and blur the background"` | **One** edit. All five words became edit targets, and because `"background and"` matched a global-hint pattern the whole thing became a *global* edit - the blur instruction vanished into the recolor. |
| `"change the shirt of the person on the left"` | `"left"` became just another edit term. Nothing resolved *which* person. |
| `"make it look better"` | Global fallback, `alignment=aligned`, `assumption=None` - no record of what "better" was taken to mean. |

## 2. Output structure

```json
{
  "prompt": "change the shirt color to red and blur the background",
  "status": "ok",
  "assumption": null,
  "clarifying_question": null,
  "instructions": [
    {
      "raw_text": "change the shirt color to red",
      "action": "recolor",
      "target": "shirt",
      "attribute": "red",
      "scope": "local",
      "position": null, "ordinal": null, "intensity": null,
      "constraints": [], "nouns": ["shirt"], "confidence": 1.0,
      "resolution": {
        "label": "shirt", "method": "explicit", "index": null,
        "confidence": 1.0, "alternatives": [], "matched_on": "shirt"
      }
    },
    { "raw_text": "blur the background", "action": "blur",
      "target": "background", "scope": "local", "...": "..." }
  ],
  "trace": ["split into 2 instruction(s): [...]", "..."]
}
```

`status` is the control signal: `ok` proceeds, `assumed` proceeds with
`assumption` recorded, `clarify` stops and returns `clarifying_question`.
`PromptIntent.should_generate` collapses that to a boolean.

`trace` is a human-readable list of every decision the stage made, so a wrong
answer can be diagnosed without a debugger.

## 3. How each piece works

**Splitting.** `split_instructions` breaks on `and` / `then` / `,` / `;`, but only
when the following clause has an action verb of its own. So `"red and blue
stripes"` stays one instruction, and a constraint clause (`"but keep the
background neutral"`) is folded into the instruction it modifies rather than
becoming a phantom edit - it is recorded in `constraints`.

**Decomposition.** `parse_instruction` matches the longest action verb phrase,
then extracts target, attribute, scope, position, ordinal and intensity. Two
details worth knowing:

* Generic verbs (`change`, `adjust`, `modify`) name no operation on their own, so
  the action is resolved from the attribute: a colour gives `recolor`, a style
  gives `restyle`, anything else `replace`.
* A possessive names the owner first and the edited thing second, so
  `"the person's shirt"` has `target="shirt"` (that is what gets masked) with
  `person` kept in `nouns` for disambiguation.

**Disambiguation.** `resolve_target` takes detector output (`SceneObject`: label,
centre, area, optional salience) and picks in this order:

1. explicit **position** in the prompt (`left`, `right`, `top`, `foreground`, ...)
2. explicit **ordinal** (`the second person`, ordered left to right)
3. **salience**, but only when the top candidate wins by `salience_margin`
   (default 0.15); `area` stands in when the detector gives no score
4. otherwise `unresolved` → the stage asks rather than guessing

The noun matched against the scene need not be the edit target: `"change the
person's shirt"` masks the *shirt* but disambiguates on *person*, which is what
`matched_on` records - and what the clarifying question names.

**Vague prompts.** `"make it look better"` grounds an assumption in image context
(`image_type`, `main_subject`) and logs it. With no context to reason from it asks
instead, honouring the "do not guess blindly" constraint.

## 4. Latency

0.24 ms per prompt (mean over 8,000 calls), which is **0.27%** of a 90 ms 20-step
denoise loop. The stage is regex and lexicon lookups - no model call - so it
cannot meaningfully move end-to-end latency.

## 5. Extending it

To add a field to `EditInstruction`:

1. Add it to the dataclass with a default, so older callers still construct.
2. Populate it in `parse_instruction`.
3. If it should affect region choice, read it in `resolve_target`.
4. If the pipeline needs it, read it from the selected instruction passed through
   `plan_edit(intent=..., instruction_index=...)`.
5. `to_dict()` picks it up automatically via `asdict`.

To add an action, add a row to `_ACTION_VERBS`; matching is longest-phrase-first,
so multi-word verbs beat single words without ordering care. New colours or styles
go in `_COLORS` / `_STYLES`. New position words go in `_POSITIONS`, mapping to one
of the canonical values `resolve_target` already handles.

## 6. Limitations

1. **Lexical, not semantic.** It matches words, so an unlisted verb falls back to
   `action="unknown"` (with reduced `confidence`, which is reported rather than
   hidden). This is the deliberate trade for the latency budget; an LLM pass would
   handle paraphrase but costs a round trip.
2. **English only.** Every lexicon and the possessive/ordinal patterns are English.
3. **Scene facts are an input.** Disambiguation needs a detector or segmenter to
   supply `SceneObject`s. Without them the stage trusts the prompt's own noun and
   never disambiguates - it does not invent candidates.
4. **Salience is approximated by area** unless a detector supplies a real score. A
   large background object can out-rank a small subject.
5. **Ordinals are ordered left-to-right only.** "the second person" from the top,
   or in reading order for a grid, is not modelled.
6. **The public edit API accepts one atomic instruction.** It rejects compound
   intents before inference because one uploaded mask cannot safely identify several
   independently targeted regions. Lower-level callers may plan a specific item with
   `instruction_index`, but composing multiple plans remains their responsibility.
