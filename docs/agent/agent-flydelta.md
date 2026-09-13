# FlyDelta sideband learning — design hypothesis

## Status and purpose

**Status: initial model-free contract slices implemented; runtime steering is
not implemented.** FlyDelta is not an active model capability or a replacement
for the current model-adaptation path. It must not be activated until it has
passed explicit evaluation and promotion gates.

FlyDelta is a proposed, small, host-controlled associative sideband for a
frozen language model. It records only host-certified experience and can
produce a bounded, reversible activation-steering overlay. The base GGUF is
never edited. It therefore complements the existing learning ladder:

```text
turn evidence
  ├─ project-/user-specific fact        -> memory
  ├─ repeatable host workflow           -> procedure / blueprint
  ├─ repeated model behavior deficiency -> corpus / LoRA candidate
  └─ repeated recognizable behavior     -> FlyDelta sideband candidate
```

The host stays authoritative. The model may generate candidates, but it
cannot certify evidence, write an active sideband, or activate one.

```text
generation != evidence
evidence   != learning update
learning   != promotion
promotion  != activation
```

The existing transaction, corpus, trainer, evaluator and LoRA-registry route
is described in [Agent model adaptation](agent-model-adaptation.md). FlyDelta
reuses its evidence and promotion principles but is not a LoRA adapter or a
training-corpus format.

The current implementation contains the model-free sparse encoder,
recognition memory, bounded delta memory, versioned JSON artifact and
compatibility checks, plus host-certified candidate, capture-manifest and
sideband-evaluation contracts. These are contract/library slices only: no
daemon, CLI or server inference path loads or applies a FlyDelta artifact yet.

## Intended mechanism

The full research hypothesis is a sparse contextual association:

```text
selected hidden state h
       |
       v
normalize -> sparse deterministic projection -> top-k winners (phi)
                                                   |
                                                   v
                                         recognition + delta memory
                                                   |
                                                   v
                                      steering coefficients alpha
                                                   |
                                                   v
                                  per-layer steering basis B_l * alpha
                                                   |
                                                   v
                                    residual addition: h'_l = h_l + delta_h
```

Conceptually, `h'_l = h_l + B_l W TopK(R Norm(h_l))`. `R` is a fixed sparse
projection, `W` a delta-rule associative memory, and `B_l` a small per-layer
steering basis. A normalized local update can be:

```text
prediction = W * phi
error      = target - prediction
W          = decay * W + learning_rate * confidence * error * phi^T
                                      / (epsilon + ||phi||^2)
```

This is a hypothesis to test, not a claim that it avoids forgetting. It must
demonstrate held-out retention and a low false-intervention rate.

### Recognition and correction are different memories

| Part | Question | Permitted result |
| --- | --- | --- |
| Recognition | “Is this context sufficiently familiar?” | familiarity, novelty, confidence |
| Correction | “Which bounded steering coefficients are relevant?” | clipped coefficients only |

Recognition is a gate, not evidence of truth. Low familiarity, high novelty,
an artifact compatibility mismatch, failed validation, or disabled policy must
all produce a no-op.

## Safety and ownership boundaries

- FlyDelta must not relax policy, confirmation, authentication, capability
  checks, required tool execution, schema validation, resource ownership, or
  other deterministic host contracts.
- It must not treat outages, malformed provider metadata, unavailable
  resources, timeouts, policy refusals, or host-contract faults as model
  behavior defects.
- Raw user content, tool secrets, credentials and unredacted hidden states
  must not silently become sideband artifacts. Existing scope, redaction,
  provenance and revocation rules are upstream requirements.
- Updates are nearline/offline. A serving session never mutates its active
  sideband. A worker produces an immutable candidate; evaluation and explicit
  host promotion precede activation.
- Activation is profile-specific, reversible, observable and kill-switchable.
  Candidate/canary/active/retired/rejected states should mirror the current
  adaptation lifecycle without sharing its LoRA type.
- V0 compatibility is exact-model only: base model, tokenizer, template,
  architecture/layout and artifact fingerprints must match. A sideband is not
  presumed portable between Q2/Q4/Q8/FP16 variants merely by family name.

## Integration seams in the current code

### Certified evidence: reuse the learning-observation boundary

`common_learning_observation` and `common_learning_transaction` already carry
host-owned scope, signal, verification, provenance, plan/tool context and
cause classification. They are the correct upstream seam. Initial qualification
should require:

```text
cause        = model_behavior
verification = host_verified or user_confirmed
scope/policy = collection permitted
evidence     = bounded, redaction-attested, reproducible
```

The sideband references observation/transaction IDs; it does not copy
arbitrary tool output into another durable store. Do not put raw activation
buffers into the generic observation contract: they are model-specific,
potentially sensitive, and need bounded artifacts and retention policy.

Contrastive steering needs a separate immutable capture manifest with the
observation ID, profile/template fingerprints, positive/negative execution
references, verifier evidence, capture/layout revision and redaction
attestation.

### Promotion: parallel sideband registry

`common_learning_adapter_registry` is LoRA-specific. FlyDelta should use a
parallel sideband registry, or a future generic overlay registry with an
explicit `kind`; it must not masquerade as a LoRA manifest.

A FlyDelta manifest needs: format/kind, ID and revision, artifact SHA-256,
base architecture and exact model fingerprint, tokenizer/template fingerprints,
inference-layout revision, layer dimensions, encoder configuration fingerprint,
strict byte/dimension bounds, source/evaluation references and lifecycle state.
The artifact must contain neither credentials nor unbounded raw conversation
content. Import must reuse the established containment, symlink, byte-bound,
hash and atomic-write discipline.

### Model profile and residency: resolve immutable metadata

`common_agent_model_profile` currently describes a base model plus LoRA
overlays, and the residency manager caches handles by profile. A FlyDelta
selection belongs beside—not inside—the LoRA list. For example, a future
profile may conceptually contain:

```json
{
  "id": "qwen-agent-canary",
  "base_model_id": "qwen-agent",
  "adapters": [{ "adapter_id": "agent-lora-v2", "scale": 0.8 }],
  "sidebands": [{
    "id": "agent-flydelta-v1",
    "kind": "flydelta",
    "mode": "canary",
    "max_scale": 0.12
  }]
}
```

This is illustrative, not a current configuration contract. Profile validation
resolves immutable artifact metadata. Per-turn coefficients and the mutable
inference context never belong in the resident model handle. The residency
cache key must include sideband revision/hash and mode.

### Current llama.cpp control-vector seam: static V0 is viable

llama.cpp exposes `llama_set_adapter_cvec(ctx, ...)`, which adds configured
control vectors at layer residual outputs. Agent CLI generation creates a
fresh `llama_context` per turn, applies overlays, then decodes. This gives a
safe experimental V0:

```text
host-side context features or embedding
       -> recognition + coefficient prediction
       -> compose cvec: delta_l = B_l * alpha
       -> create fresh llama context
       -> set cvec before first decode
       -> normal prompt and generation
```

V0 is **context-selected static steering**, not the full hidden-state loop.
It does not read an intermediate activation and alter the same decode pass. It
is still sufficient to validate artifact compatibility, gating, composition,
rollback, latency and interference.

The cvec must be set before prompt evaluation on a fresh context. Changing it
after prompt tokens are in the KV cache would mix unsteered and steered state;
that is not valid V0 behavior. Contexts must not be shared across cvec values.

### Dynamic hidden-state capture/injection is a later seam

The public API has output embeddings and an evaluation callback; internal
graphs label layer outputs such as `l_out`. The scheduler callback is an
observation/debug facility, may transfer backend tensors to host memory, and
is not a safe public dynamic-intervention API.

The full loop needs an explicitly staged route:

1. **V1 two-pass experiment:** bounded capture/prefill, derive a code,
   discard that context, compose cvec and run a second fresh generation. This
   avoids mixed KV state but doubles prompt work.
2. **V2 optional llama.cpp hook:** a versioned, device-aware activation
   observer/intervention contract at documented residual slots. Begin CPU-only;
   unsupported backends must no-op. GPU support needs kernels and scheduling,
   not per-layer host copies.

Neither route belongs in generic inference until static V0 shows useful,
bounded results.

## Recommended implementation sweeps

### 1. Model-free associative core — implemented

Add `common/agent/adaptation/flydelta/` with a deterministic sparse encoder,
bounded delta/recognition memory, novelty scorer, artifact codec and strict
compatibility validator. Use synthetic vectors only. Tests cover determinism,
tie breaking, update/prediction, clipping, novelty no-op, tampering and atomic
candidate snapshots.

### 2. Host-certified candidate and evaluation contracts — implemented

Build sideband candidates from qualified existing transactions and separate
capture manifests. Prove with fixtures that host-contract, policy, resource and
transport failures do not qualify. Measure verified success, held-out
retention, regression/interference, false intervention, no-op rate, runtime
intervention rate, latency/token and overlay bytes per improvement.

### 3. CPU-only static cvec V0

Insert an agent-owned overlay applicator between profile resolution and the
first `llama_decode` in the CLI inference path. It validates the selected
artifact, derives host-side features, composes a cvec, applies it to a new
context and fails closed to no-op. Restrict it to one CPU architecture/path.
Server-context, residency reuse, CUDA/Vulkan, Android and dynamic capture stay
disabled until independently tested.

### 4. Contrastive basis and bounded two-pass experiment

Define exact positive/negative prompt alignment before deriving any basis.
Never subtract unrelated free-form outputs. If needed, add the bounded
two-pass capture path with explicit token/layer/byte/latency budgets.

### 5. Optional dynamic hook

Only propose an upstream-quality llama.cpp hook if the evidence warrants it.
It needs backend behavior, tensor lifetime, threading, architecture coverage,
abort/error semantics and a zero-overhead disabled path.

## First evaluation

Start with one narrow, host-verifiable behavior: a redacted structured-tool or
code-repair fixture, not free-form factual knowledge or policy behavior.

```text
A  base model
B  existing memory/procedure assistance
C  static contrastive steering
D  recognition-gated FlyDelta sideband
```

Use fixed fixtures, baseline replay and held-out evaluation. Promotion requires
an improvement without material regression; a successful worker run or lower
internal loss is not enough.

## Open decisions

1. V0 context signal: host features, an embedding model, or two-pass
   model-derived features. The first two are cheaper and safer.
2. Initial domain: coding repair or structured tool behavior. Tool contracts
   remain host-enforced even if a sideband improves model choices.
3. Registry shape: a sideband-specific registry now, or a deliberately planned
   generic overlay registry later.
4. Exact redaction, scope, TTL and revocation semantics for capture manifests.

Until these are decided, documentation, model-free experiments and strict
non-activation are the correct next steps—not an automatic learning loop.
