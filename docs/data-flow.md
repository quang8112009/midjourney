# Data flow

How a request travels through the system, what transforms it, and where the
current architecture diverges from that.

## 1. Live flows

Two request paths reach real code. Both are healthy; the issues in §3 are
elsewhere.

### Generation

```
frontend/index.html                      POST /api/v1/generate
        |
        v
GenerateRequest (pydantic)               routes.py:39
  - normalises prompt / negative_prompt  (field_validator)
  - resolves per-model defaults          (model_validator)  <-- also done in the service
  - enforces size, aspect, steps, seed   -> HTTP 422 on violation
        |
        v
execute_generation()                     routes.py:225
  - optional enhance_prompt(seed)        prompt_enhancer.py   [transform]
  - run_in_threadpool(generate_image)    keeps the event loop free
        |
        v
ModelService.generate_image()            model_service.py:240
  - _validate_generation_args()          defence in depth for direct callers
  - lazy load_model() under a lock       single pipeline in memory
  - torch.inference_mode()               [the expensive step]
        |
        v
save_images() -> outputs/*.png           atomic write + LRU prune  [persist]
        |
        v
GenerateResponse                         id, images[], parameters, effective_prompt
```

### Chat

```
frontend            POST /api/v1/chat
        |
        v
ChatRequest -> ChatService.handle_turn()          per-session asyncio lock
        |
        v
ReasoningService.reason()                          two provider calls
  pass 1: REASONING_SYSTEM_PROMPT -> <reasoning> XML -> ReasoningAnalysis
  pass 2: analysis rendered into `instructions`  -> public reply
        |
        v (only when action == generate_image)
GenerateRequest(prompt=resolved, **client_options)  re-validated, chat_routes.py:162
        |
        v
execute_generation() ... same path as above
        |
        v
ConversationStore.append_exchange()                 atomic commit  [persist]
        |
        v
ChatResponse            message, session_id, turn_id, trace_id, generation?
```

**Verified healthy:**

* **Contracts match across the boundary.** Every field the frontend sends exists
  on the pydantic model, and every field it reads exists on the response. No
  casing or naming drift.
* **Layered validation does not drift.** The API validator and
  `_validate_generation_args` were probed against six boundary cases (extreme
  aspect, oversize, PixArt non-512, step/guidance limits, oversized prompt) and
  agreed on every one. The duplication is deliberate: the API returns 422, the
  service protects direct callers.
* **The model's own output is re-validated.** A prompt the LLM resolves is rebuilt
  into a `GenerateRequest` before generation, so it passes the same gate as any
  client request. That is a safety boundary, not a redundant hop.

## 2. The orphaned island

```
   app/services/editing/   (9 modules, 4,140 lines, 184 tests)
   prompt_intent · semantic_planner · edit_planner · layout_guidance
   edit_pipeline · region_attention · adaptive_reference · masks · metrics
                              |
                              X   no import from app/api, app/core, or main
                              |
                        (unreachable)
```

Nothing in `app/api/`, `app/core/`, or `main.py` imports the editing package. It
is a fully-tested subsystem with no entry point: the flow above never touches it.

## 3. Issues, by impact

| # | Issue | Where | Impact |
|---|---|---|---|
| 1 | Editing package unreachable from any route | `app/services/editing/*` | 4,140 lines, 184 tests, no request can run it |
| 2 | Three prompt parsers, disagreeing | `prompt_intent`, `semantic_planner`, `edit_planner` | 4 of 6 sample prompts produce different target/action |
| 3 | Attribute regex swallowed clauses | `semantic_planner` | *fixed* - see §4 |
| 4 | Per-model defaults resolved twice | `routes.py:80`, `model_service.py:250` | consistent today; two copies to keep in sync |

### Issue 2 in detail

For `"change the shirt to red and blur the background"`:

| parser | result |
|---|---|
| `prompt_intent.analyze_prompt` | 2 instructions: recolor/shirt/red, blur/background |
| `semantic_planner.plan_semantic_layout` | 1 edit_target: recolor/**background**/red |
| `edit_planner.select_edit_terms` | flat terms, no structure |

They disagree on target, on how many edits exist, and on scope. Whichever the
pipeline eventually consumes decides the edit, so today there is no single answer
to "what did the user ask for".

## 4. Applied

`semantic_planner` attribute extraction anchored its non-greedy capture to `$`,
so it consumed the rest of the sentence:

```
"change the shirt to red and blur the background"
  before -> attribute = "red and blur the background"
  after  -> attribute = "red"

"restyle it into a watercolor painting and brighten it"
  before -> attribute = "watercolor painting and brighten it"
  after  -> attribute = "watercolor painting"
```

Fixed by stopping the capture at a clause boundary (`and|then|but|while|,|;`).
272 tests still pass.

## 5. Proposed target flow

One parser, one contract, one entry point:

```
prompt --> analyze_prompt() --> PromptIntent (JSON)   <-- single source of truth
                                     |
              +----------------------+----------------------+
              |                                             |
        plan_edit(instruction=...)                 plan_semantic_layout(intent=...)
        region + coefficients                      object layout + boxes
              |                                             |
              +----------------------+----------------------+
                                     v
                        run_region_aware_edit()  /  hybrid edit
                                     v
                              POST /api/v1/edit
```

`PromptIntent` already exists and is JSON-serialisable, so it can be the contract
between the understanding stage and both planners. `semantic_planner` would keep
layout (boxes, relations, counts) and drop its own action/target/attribute
extraction, deleting the duplicate parser rather than reconciling it.

## 6. Not applied - needs a decision

1. **Wire the editing package to a route** (`POST /api/v1/edit` taking a source
   image + mask). This is the change that makes issue 1 real work rather than
   dead code, and it is a new public endpoint - an API surface decision.
2. **Collapse the three parsers onto `PromptIntent`.** Structural: it changes
   `plan_semantic_layout`'s output contract and touches `test_hybrid_reasoning`.
3. **Single-source the per-model defaults** so `routes.py` and `model_service`
   share one resolver. Low risk, but it moves behaviour between layers.
