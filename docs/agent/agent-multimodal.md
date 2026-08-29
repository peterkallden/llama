# Agent multimodal runtime

Multimodal support is being added as one vertical agent-runtime path. The
agent owns resource identity, policy, and fallback decisions; llama.cpp keeps
ownership of model loading and native multimodal execution.

The model-profile and multi-model lifetime rules are documented in
[Agent model residency and multi-model scheduling](agent-model-residency.md).
An `mmproj` belongs to the selected model/load identity. It is not a free
per-turn file and it must not be reused across profiles with different model,
tokenizer or template identity.

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
backend seam. When model profiles are served from one process, each backend
loader must retain this rule during profile resolution, before model loading.

## Current staged scope

Configuration, session identity, capability contracts, and host-owned resource
resolution are implemented. Image and audio references are passed through the
same generation contract to the server-context adapter. Existing OCR/page-
image processors remain the fallback path for models without native image
support.

For an image without native vision capability, the server-context adapter now
asks the host-owned processing provider for a `text` representation. With the
`ocr.tesseract` policy enabled, `image/*` resources resolve through the
existing Tesseract processor and the derived text is added to the text prompt.
Native vision remains preferred. Audio has no equivalent OCR fallback and is
reported as unsupported/not-run unless the model natively supports audio.

The existing text path remains unchanged and is the compatibility baseline for
models such as Qwen text-only checkpoints. Model-backed Qwen verification will
continue to exercise that baseline; multimodal cases must be marked not-run
when the selected model has no projector or image capability.

The model-backed fixture set includes `cats.jpg`, `scb-cpi.png`, and
`sample-speech.mp3`. The native smoke executes image cases and an audio case;
an image-only projector reports the audio case as `not-run`, while malformed
or incorrectly resolved audio remains a failure. The resource CTest contract
also carries image and audio references together, so audio does not introduce
a second resource or generation path.
