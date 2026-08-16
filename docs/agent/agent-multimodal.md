# Agent multimodal runtime

Multimodal support is being added as one vertical agent-runtime path. The
agent owns resource identity, policy, and fallback decisions; llama.cpp keeps
ownership of model loading and native multimodal execution.

## Current contract

The host model configuration may contain an optional `mmproj` path next to the
GGUF model path:

```json
{
  "model": {
    "path": "models/model.gguf",
    "mmproj": "models/model-mmproj.gguf"
  }
}
```

The path is carried through host configuration, daemon options, resident
requests, inference options, and model-load identity. This prevents a text
model and a model with a different projector from being accidentally reused in
the same resident session.

The `server-context` backend reports text, image, and audio capability from its
loaded model metadata. The CLI backend remains text-only for now. Supplying
`mmproj` to the CLI backend is rejected explicitly instead of silently
ignoring the projector; native multimodal execution therefore has one clear
backend seam.

## Scope of the first sweep

This sweep establishes configuration, session identity, and capability
contracts. It does not yet attach agent resources to inference messages. That
is the next sweep: resource references will be resolved by the host and passed
to the server-context adapter, with existing OCR/page-image processors kept as
fallbacks for models without native image support.

The existing text path remains unchanged and is the compatibility baseline for
models such as Qwen text-only checkpoints. Model-backed Qwen verification will
continue to exercise that baseline; multimodal cases must be marked not-run
when the selected model has no projector or image capability.
