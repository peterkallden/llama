# llama.cpp/tools/imatrix

Compute an importance matrix for a model and given text dataset. Can be used during quantization to enhance the quality of the quantized models.
More information is available in <https://github.com/ggml-org/llama.cpp/pull/4861>.

## Usage

```
./llama-imatrix \
    -m model.gguf -f some-text.txt [-o imatrix.gguf] [--output-format {gguf,dat}] [--no-ppl] \
    [--process-output] [--chunk 123] [--save-frequency 0] [--output-frequency 10] \
    [--in-file imatrix-prev-0.gguf --in-file imatrix-prev-1.gguf ...] [--parse-special] \
    [--show-statistics] [...]
```

Here `-m | --model` with a model name and `-f | --file` with a file containing calibration data (such as e.g. `wiki.train.raw`) are mandatory.
The parameters in square brackets are optional and have the following meaning:

* `-h | --help` shows usage information and exits.
* `-lv | --verbosity` specifies the verbosity level. If set to `0`, no output other than the perplexity of the processed chunks will be generated. If set to `1`, each time the results are saved a message is written to `stderr`. If `>=2`, a message is output each time data is collected for any tensor. Default verbosity level is `1`.
* `-o | --output-file` specifies the name of the file where the computed data will be stored. If missing `imatrix.gguf` is used.
* `-ofreq | --output-frequency` specifies how often the so far computed result is saved to disk. Default is 10 (i.e., every 10 chunks)
* `--output-format` specifies the output format of the generated imatrix file. Either "gguf", or "dat" (the legacy format). Defaults to "gguf".
* `--save-frequency` specifies how often to save a copy of the imatrix in a separate file. Default is 0 (i.e., never)
* `--process-output` specifies if data will be collected for the `output.weight` tensor. Typically, it is better not to utilize the importance matrix when quantizing `output.weight`, so this is set to `false` by default.
* `--in-file` one or more existing imatrix files to load and combine. Useful for merging files from multiple runs/datasets.
* `--parse-special` enables parsing of special tokens (e.g., `<|im_start|>` in some models). Useful for models with custom tokenizers.
* `--chunk | --from-chunk` to skip the first `n` chunks of tokens from the input data. Useful for resuming or skipping initial low-quality data.
* `--chunks` maximum number of chunks to process. Default is -1 for all available chunks.
* `--no-ppl` disables the calculation of perplexity for the processed chunks. Useful if you want to speed up the processing and do not care about perplexity.
* `--show-statistics` displays imatrix file's statistics.
* `--collect-quant-profile` stores optional sampled rate-distortion curves, block statistics, and sampled adjacent-layer delta metrics in the GGUF imatrix. Existing imatrix consumers ignore these namespaced tensors.
* `--quant-profile-sample-rows` controls sampled weight rows per tensor for the optional profile. Default is 8.
* `--quant-profile-block-size` controls the number of imatrix values summarized per optional analysis block. Default is 256.
* `--quant-profile-refine-top-k` enables high-fidelity RD refinement for the top-K hotspot tensors. Default is 0, disabled.
* `--quant-profile-refine-rows` controls retained and evaluated weight rows for refined tensors. Default is 32 and must exceed `--quant-profile-sample-rows`.

For faster computation, make sure to use GPU offloading via the `-ngl | --n-gpu-layers` argument.

Recent versions of `llama-imatrix` store data in GGUF format by default. For the legacy format, use an extension other than `.gguf` when saving the output file. More information is available in <https://github.com/ggml-org/llama.cpp/pull/9400>.

## Examples

```bash
# generate importance matrix using default filename (imatrix.gguf), offloading 99 layers to GPU
./llama-imatrix -m ggml-model-f16.gguf -f calibration-data.txt -ngl 99

# use the imatrix to perform a Q4_K_M quantization
./llama-quantize --imatrix imatrix.gguf ggml-model-f16.gguf ./ggml-model-q4_k_m.gguf q4_k_m
```

```bash
# generate a reusable quantization-analysis profile alongside the normal imatrix
./llama-imatrix -m ggml-model-f16.gguf -f calibration-data.txt \
  --collect-quant-profile --quant-profile-sample-rows 8 \
  --quant-profile-refine-top-k 16 --quant-profile-refine-rows 32 \
  -o imatrix-profile.gguf -ngl 99

# spqr-layer-delta and rd-guided reuse compatible profile entries automatically
./llama-quantize --imatrix imatrix-profile.gguf --mixed-policy spqr_layer_delta \
  --adaptive-anchors --rd-guided ggml-model-f16.gguf model-guided.gguf q3_k_m
```

The profile is optional and policy-neutral: it stores raw candidate distortion and similarity metrics, not selected quantization types or sensitivity buckets. It also stores activity-mask statistics per tensor: mean squared activity, variance across channels, peak-to-mean ratio, and active-channel fraction. RD profile reuse currently covers `Q3_K`, `Q4_K`, `Q5_K`, and `Q6_K`. Profile generation from F16, BF16, or F32 model weights is supported; other source weight types are skipped.

Quantization analysis profile version 5 declares its available features in `imatrix.analysis.features`, including `block_rd` and `rd_refinement`. Deterministically selected weight rows act as analysis blocks; per-candidate distortion combines mean, p90, and worst-block distortion. Q3 and Q5 are always evaluated during the coarse pass, while Q4 and Q6 are evaluated adaptively. When refinement is enabled, tensors are ranked by tail distortion, Q3-to-Q5 curve gain, and activity risk. The top-K tensors are then reevaluated over more retained rows with all Q3/Q4/Q5/Q6 candidates. This increases temporary imatrix memory use but avoids a second model pass. `llama-quantize` validates and reports refined profile entries automatically. Version 2 through 4 profiles remain readable.

The guided quantizer converts activity statistics into a bounded rare-event risk multiplier for RD distortion. Adaptive anchors combine adjacent-layer weight delta with changes in activity statistics and use a conservative robust-MAD scene-change threshold. These choices remain quantization-time policy decisions; the stored profile contains only reusable raw statistics.

RD-guided quantization can optionally use `--rd-target-bpw` or `--rd-target-size-mib` as a soft global model-size target. The profile's candidate curves are allocated globally, while `--rd-lambda` limits compression pressure. If reaching the target would exceed that quality limit, the quantizer keeps the larger model and reports the difference.

To avoid repeating candidate evaluation during calibration, periodic and snapshot imatrix saves omit the optional profile. The final output written when calibration completes contains it.

```bash
# generate and save the imatrix using legacy format
./llama-imatrix -m ggml-model-f16.gguf -f calibration-data.txt --output-format dat -o imatrix-legcy-format.dat -ngl 99
```

```bash
# convert legacy (binary) imatrix format to new (GGUF) format
./llama-imatrix --in-file imatrix-legacy-format.dat -o imatrix-new-format.gguf
```

```bash
# convert new (GGUF) imatrix format to legacy (binary) format
./llama-imatrix --in-file imatrix-new-format.gguf --output-format dat -o imatrix-legacy-format.dat
```

```bash
# combine existing imatrices
./llama-imatrix --in-file imatrix-prev-0.gguf --in-file imatrix-prev-1.gguf -o imatrix-combined.gguf
```

```bash
# skip first 5 chunks, save intermediates every 20 chunks and snapshots every 50, parsing special tokens
./llama-imatrix -m ggml-model-f16.gguf -f calibration-data.txt --chunk 5 --output-frequency 20 --save-frequency 50 --parse-special
```

```bash
# analyse imatrix file and display summary statistics instead of running inference
./llama-imatrix --in-file imatrix.gguf --show-statistics
```

`--show-statistics` will display the following statistics:

#### Per tensor

* Σ(Act²): sum of all squared activations (the importance scores)
* Min & Max: minimum and maximum squared activations values
* μ & σ: Squared activations' mean and standard deviation
* % Active: proportion of elements whose average squared activation exceeds a small threshold (1e-5). Helpful to determine how alive/dormant the tensor is during inference
* N: number of squared activations
* Entropy: entropy of the squared activation distribution, in bits (standard Shannon entropy measurement) $S = -\sum_{i=1}^N p_i \log_2 p_i$
* E (norm): Normalized entropy. $E(norm)=\frac{-\sum_{i=1}^N p_i \log_2 p_i}{log_2 N}$. These two metrics can be used to determine how well a prompt "exercises" the model's capabilities
* ZD Score: z-score distribution as described in _3.1 Layer Importance Scores_ of [Layer-Wise Quantization](https://arxiv.org/abs/2406.17415)
* CosSim: cosine similarity with respect to the previous layer's tensor. Useful to determine how similar the squared activations of the current layer are to the previous layer's squared activations.

#### Per layer

Weighted averages of Σ(Act²), ZD Score and CosSim are also calculated.

#### Important note on the computed Statistics

When using these statistics, please note that they are computed on the squared activations, **not on the actual (raw) activations**.
Whilst the results are still useful, they're less reliable than using the raw values, and in the case of the cosine similarity, could be misleading if the tensor contains opposite vectors.
