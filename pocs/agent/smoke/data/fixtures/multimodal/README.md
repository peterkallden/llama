# Multimodal smoke fixtures

These fixtures are used by model-backed multimodal and OCR/fallback smokes.

- `cats.jpg` — visual scene containing cats; native multimodal models should
  identify the visual content. OCR is not expected to produce a useful result.
- `scb-cpi.png` — SCB CPI statistics image; native multimodal models should
  describe the chart/text, while OCR fallback can validate the textual
  representation.

Expected capability-aware results:

| Runtime/model capability | `cats.jpg` | `scb-cpi.png` |
| --- | --- | --- |
| Native image support | pass | pass |
| Text-only model with OCR fallback | not-run for visual understanding | pass |
| Text-only model without fallback | not-run | not-run |

Missing capability is an expected `not-run`/skip condition, not a failed test.

The model-backed native smoke is run manually or from a model-enabled job:

```bash
scripts/test-agent-multimodal-smoke.sh \
  build-agent-packaging \
  /path/to/model.gguf \
  /path/to/mmproj.gguf
```

It returns exit code `77` when the agent binary, model, or projector is not
available, so a CI wrapper can report the case as not-run.
Malformed, unreadable, or incorrectly resolved fixtures remain failures.
