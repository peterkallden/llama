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

After selecting candidate features in `circuit-tracer`, append measurements:

```sh
python pocs/circuit-transfer/circuit_transfer.py record \
  --run-file work/circuit-transfer/results.jsonl \
  --case-id capital-france \
  --intervention inject-to-restore \
  --site blocks.12.feature.1234 \
  --baseline-logit-diff -1.2 \
  --intervention-logit-diff 3.7 \
  --notes "Candidate restored the target on the corrupt prompt"
```

Summarize recorded measurements:

```sh
python pocs/circuit-transfer/circuit_transfer.py summarize \
  --run-file work/circuit-transfer/results.jsonl
```

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

Useful upstream projects:

- `decoderesearch/circuit-tracer`
- `OpenMOSS/Llamascopium`
- `fnlp/Llama-Scope` on Hugging Face

