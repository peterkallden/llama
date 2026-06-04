# Circuit transfer proof of concept

This directory is an experimental workbench for circuit discovery and soft
circuit transfer on Llama-family models. It intentionally lives outside the
`llama.cpp` runtime: discovery is performed with PyTorch-based tools, while
`llama.cpp` becomes useful later for GGUF LoRA conversion and regression tests.

The first milestone is deliberately small:

1. Define contrastive prompt pairs for a measurable behavior.
2. Generate reproducible `circuit-tracer` attribution commands.
3. Discover candidate features in `Llama-3.2-1B`.
4. Verify candidates with ablate-to-fail and inject-to-restore interventions.
5. Save measurements in a stable JSONL format.

Raw attention-head transplantation is out of scope for this milestone. Llama
models often distribute a behavior across positions and components, and Llama
3.x grouped-query attention requires architecture-aware slicing.

## Setup

Create a Python environment for the external research tools. Keep it separate
from the Python environment used by the conversion scripts in the repository.

```sh
python -m venv .venv
. .venv/bin/activate
pip install circuit-tracer
```

`circuit-tracer` downloads model and transcoder weights from Hugging Face on
first use. Access to gated Meta Llama weights may require accepting the model
license and authenticating with Hugging Face.

## Validate the prompt set

```sh
python pocs/circuit-transfer/circuit_transfer.py validate
```

The starter dataset uses factual recall because target and distractor logits
are easy to compare. Replace or extend it before drawing conclusions.

## Draft dynamic cases

The starter cases are intentionally small. To explore a different circuit, draft
a new case file from a context or theme.

Generate a prompt for a source/teacher model:

```sh
python pocs/circuit-transfer/case_draft.py prompt \
  --context "basic astronomy facts with clean country/planet contrasts" \
  --count 8 \
  --output work/circuit-transfer/cases/teacher-prompt.txt
```

Ask your source model with that prompt, save the JSONL answer, and normalize it:

```sh
python pocs/circuit-transfer/case_draft.py from-output \
  --input work/circuit-transfer/cases/teacher-output.jsonl \
  --output work/circuit-transfer/cases/astronomy.jsonl
```

If the source model is available through Hugging Face locally, you can combine
both steps:

```sh
python pocs/circuit-transfer/case_draft.py generate-with-model \
  --model path-or-hf-model-id \
  --context "basic astronomy facts with clean planet contrasts" \
  --output work/circuit-transfer/cases/astronomy.jsonl \
  --raw-output work/circuit-transfer/cases/astronomy.raw.txt
```

Check whether target and distractor are single tokens for the model you will
score:

```sh
python pocs/circuit-transfer/case_draft.py check-tokenization \
  --cases work/circuit-transfer/cases/astronomy.jsonl \
  --model path-or-hf-model-id
```

Then pass the case file into existing commands:

```sh
python pocs/circuit-transfer/circuit_transfer.py render-tracer \
  --cases work/circuit-transfer/cases/astronomy.jsonl \
  --output-dir work/circuit-transfer/graphs
```

## Generate attribution commands

```sh
python pocs/circuit-transfer/circuit_transfer.py render-tracer \
  --output-dir work/circuit-transfer/graphs
```

Run one rendered command to create a graph:

```sh
circuit-tracer attribute \
  --prompt "The capital of France is" \
  --transcoder_set llama \
  --slug capital-france-clean \
  --graph_file_dir work/circuit-transfer/graphs/capital-france-clean \
  --graph_output_path work/circuit-transfer/graphs/capital-france-clean.pt
```

The `llama` preset currently targets `Llama-3.2-1B` with published transcoders.
Use `--server` on an individual command when interactive graph inspection is
useful.

## Record an intervention

Result rows follow `results.schema.json`. The key fields are the tested model,
prompt, target/distractor tokens, candidate feature site, intervention type and
before/after logit difference.

After selecting candidate features in `circuit-tracer`, append measurements:

```sh
python pocs/circuit-transfer/circuit_transfer.py record \
  --run-file work/circuit-transfer/results.jsonl \
  --case-id capital-france \
  --prompt "The capital of Germany is" \
  --target " Paris" \
  --distractor " Berlin" \
  --intervention inject-to-restore \
  --layer 12 \
  --feature-id 1234 \
  --baseline-logit-diff -1.2 \
  --intervention-logit-diff 3.7 \
  --notes "Candidate restored the target on the corrupt prompt"
```

Summarize recorded measurements:

```sh
python pocs/circuit-transfer/circuit_transfer.py summarize \
  --run-file work/circuit-transfer/results.jsonl
```

Classify whether the current measurements look like a probable success:

```sh
python pocs/circuit-transfer/circuit_transfer.py evaluate-success \
  --run-file work/circuit-transfer/results.jsonl
```

The default success criteria are intentionally conservative:

- mean `inject-to-restore` delta is at least `+2.0`;
- mean `ablate-to-fail` delta is at most `-1.0`;
- the effect appears across at least two prompt templates;
- unrelated or wrong-site controls stay within `0.5` absolute logit-diff delta.

The possible statuses are:

- `probable_success`: target effects, prompt coverage and controls pass;
- `promising_needs_controls`: target effects and prompt coverage pass, but no controls are logged;
- `partial_signal`: at least one target-side effect is present, but the package is incomplete;
- `failed_or_insufficient`: no clear target-side effect yet.

## Build a verification plan

Capture candidate features from attribution graph inspection in a JSONL file:

```json
{"case_id":"capital-france","layer":12,"feature_id":1234,"graph_path":"work/circuit-transfer/graphs/capital-france-clean.pt","notes":"Candidate from graph inspection"}
```

Then generate the two checks for each candidate:

```sh
python pocs/circuit-transfer/verify_interventions.py \
  --cases work/circuit-transfer/cases/astronomy.jsonl \
  --candidates pocs/circuit-transfer/data/candidate_features.example.jsonl \
  --output work/circuit-transfer/verification-plan.json
```

Each candidate gets:

- `ablate-to-fail` on the clean prompt;
- `inject-to-restore` on the corrupt prompt.

The script does not call `circuit-tracer` directly yet. It produces a stable
plan that can be executed manually or wired into a future runner once the
research environment is installed.

## Run a steering injection

The first runtime test is soft steering. Instead of editing weights, inject a
candidate vector into the residual stream and measure whether the target logit
beats the distractor more strongly.

Use a vector JSON file:

```json
{
  "layer": 12,
  "feature_id": 1234,
  "source": "circuit-tracer candidate",
  "values": [0.1, -0.2, 0.05]
}
```

The example vector in `data/steering_vector.example.json` is only a tiny
placeholder for dry-run validation. Replace it with a real residual or decoder
vector before interpreting results.

Normalize an externally exported vector into the expected format:

```sh
python pocs/circuit-transfer/vector_io.py from-json \
  --input work/circuit-transfer/raw-vector.json \
  --output work/circuit-transfer/vectors/capital-france-feature-1234.json \
  --layer 12 \
  --feature-id 1234
```

Validate the plan and vector without loading a model:

```sh
python pocs/circuit-transfer/steering_injection.py dry-run \
  --plan work/circuit-transfer/verification-plan.json \
  --vector-file pocs/circuit-transfer/data/steering_vector.example.json
```

Run the actual Hugging Face intervention:

```sh
python pocs/circuit-transfer/steering_injection.py run \
  --plan work/circuit-transfer/verification-plan.json \
  --vector-file work/circuit-transfer/vectors/capital-france-feature-1234.json \
  --run-file work/circuit-transfer/results.jsonl \
  --model meta-llama/Llama-3.2-1B \
  --strength 1.0 \
  --top-k 20
```

For `inject-to-restore`, the hook adds `strength * vector` at the selected layer
and token position. For `ablate-to-fail`, it removes the hidden state's
projection along the same vector. The default token position is the final prompt
token (`-1`), which is the first useful target for next-token factual recall.
When `--top-k` is greater than zero, each result row also records baseline and
intervention top-k token distributions. Those become the soft teacher signal
for LoRA distillation.

After collecting rows, run:

```sh
python pocs/circuit-transfer/circuit_transfer.py evaluate-success \
  --run-file work/circuit-transfer/results.jsonl
```

## Distill to a small LoRA

When steering gives a `probable_success` or a strong
`promising_needs_controls`, distill the effect into a small LoRA. The first
version uses successful `inject-to-restore` rows as target-token supervision.
That is a simple proxy for the teacher intervention; full logit-distribution
distillation can be added after the runner records teacher logits.

Prepare the dataset:

```sh
python pocs/circuit-transfer/distill_lora.py prepare-dataset \
  --run-file work/circuit-transfer/results.jsonl \
  --output work/circuit-transfer/distill/train.jsonl \
  --min-delta 2.0
```

Dry-run the LoRA config:

```sh
python pocs/circuit-transfer/distill_lora.py dry-run \
  --dataset work/circuit-transfer/distill/train.jsonl \
  --rank 4 \
  --target-modules q_proj,v_proj
```

Train a small PEFT LoRA:

```sh
python pocs/circuit-transfer/distill_lora.py train-lora \
  --dataset work/circuit-transfer/distill/train.jsonl \
  --output-dir work/circuit-transfer/lora/capital-france-r4 \
  --model meta-llama/Llama-3.2-1B \
  --rank 4 \
  --target-modules q_proj,v_proj \
  --steps 100 \
  --learning-rate 1e-4 \
  --top-k-kl-weight 0.2 \
  --temperature 1.0
```

Then evaluate `base + LoRA` with the same prompts and success criteria. Only
convert the adapter to GGUF if the LoRA approximates the steering effect without
breaking controls:

```sh
python pocs/circuit-transfer/evaluate_lora.py run \
  --plan work/circuit-transfer/verification-plan.json \
  --adapter work/circuit-transfer/lora/capital-france-r4 \
  --run-file work/circuit-transfer/lora-results.jsonl \
  --model meta-llama/Llama-3.2-1B \
  --top-k 20
```

```sh
python convert_lora_to_gguf.py \
  --base models/llama-3.2-1b \
  --outfile work/circuit-transfer/lora/capital-france-r4.gguf \
  work/circuit-transfer/lora/capital-france-r4
```

## End-to-end plan

Generate an executable command plan for the whole PoC:

```sh
python pocs/circuit-transfer/run_pipeline.py plan \
  --output work/circuit-transfer/pipeline-plan.json \
  --work-dir work/circuit-transfer
```

Run only the light validation steps:

```sh
python pocs/circuit-transfer/run_pipeline.py dry-run \
  --work-dir work/circuit-transfer
```

The heavy steps remain explicit: steering with model weights, LoRA training and
LoRA evaluation.

## Exit criteria for milestone 1

Treat discovery as successful only when at least one candidate:

- decreases the target logit difference when ablated on a clean prompt;
- restores the target logit difference when injected into a corrupt prompt;
- repeats across more than one prompt template;
- does not broadly improve unrelated prompts.

## Next milestone: soft transfer

Once discovery is repeatable, add a PyTorch experiment that:

1. collects paired donor and recipient residual activations;
2. fits an affine map between matching residual-stream layers;
3. maps a verified donor steering vector into the recipient;
4. measures target restoration and unrelated-task regressions;
5. optionally distills the intervention into a LoRA adapter;
6. converts the LoRA adapter with `convert_lora_to_gguf.py`;
7. regression-tests the GGUF adapter with `llama-server`.

## Affine stitch transfer

The first transfer experiment maps a donor steering direction into a related
recipient model by fitting an affine map between residual streams. Start with
models that share tokenizer and hidden size.

Validate the setup:

```sh
python pocs/circuit-transfer/affine_stitch.py dry-run \
  --donor-model meta-llama/Llama-3.1-8B-Instruct \
  --recipient-model meta-llama/Llama-3.1-8B \
  --prompts pocs/circuit-transfer/data/paired_prompts.example.jsonl \
  --layers 8,12,16 \
  --output work/circuit-transfer/affine/activations.pt
```

Collect paired last-token residual activations:

```sh
python pocs/circuit-transfer/affine_stitch.py collect-activations \
  --donor-model meta-llama/Llama-3.1-8B-Instruct \
  --recipient-model meta-llama/Llama-3.1-8B \
  --prompts pocs/circuit-transfer/data/paired_prompts.example.jsonl \
  --layers 12 \
  --output work/circuit-transfer/affine/activations.pt
```

Fit a ridge affine stitch for one layer:

```sh
python pocs/circuit-transfer/affine_stitch.py fit \
  --activations work/circuit-transfer/affine/activations.pt \
  --layer 12 \
  --output work/circuit-transfer/affine/layer-12-stitch.json \
  --ridge-lambda 0.01
```

Map a verified donor vector into recipient space:

```sh
python pocs/circuit-transfer/affine_stitch.py map-vector \
  --stitch work/circuit-transfer/affine/layer-12-stitch.json \
  --vector-file work/circuit-transfer/vectors/donor-feature-1234.json \
  --output work/circuit-transfer/vectors/recipient-feature-1234.json
```

Generate the recipient evaluation commands:

```sh
python pocs/circuit-transfer/affine_stitch.py evaluate-plan \
  --stitch work/circuit-transfer/affine/layer-12-stitch.json \
  --vector-file work/circuit-transfer/vectors/donor-feature-1234.json \
  --mapped-vector work/circuit-transfer/vectors/recipient-feature-1234.json \
  --verification-plan work/circuit-transfer/verification-plan.json \
  --run-file work/circuit-transfer/recipient-steering-results.jsonl \
  --recipient-model meta-llama/Llama-3.1-8B \
  --output work/circuit-transfer/affine/evaluate-recipient-plan.json
```

Useful upstream projects:

- `decoderesearch/circuit-tracer`
- `OpenMOSS/Llamascopium`
- `fnlp/Llama-Scope` on Hugging Face
