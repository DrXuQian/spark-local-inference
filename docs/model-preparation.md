# Model preparation and serving

## Runtime packages

Run commands from the repository root. Ignition prepares a source checkpoint
into a **runtime package** and publishes a directory entry in
`paths.package_cache` (default `~/.cache/flare/packages`). Its primary,
`model.safetensors`, contains the signed `flare.manifest`, unsigned
`flare.locations`, optional reconstruction hints, and frontend assets. Readably
named safetensors companions hold layer weights, tuning-dependent weights,
initial state, zstd-compressed cubins, and graph records. Cache companions hardlink
immutable payload objects whose identities hash the payload bytes.

The first canonical form packs the package into one `.safetensors` file; the
second is the directory form used by the cache. Export always publishes to that
cache. `ignition convert CACHE_ENTRY --output DIRECTORY` creates a standalone
directory; `ignition convert CACHE_ENTRY --form packed --output FILE` creates
one exact file path. `CACHE_ENTRY` is the path printed by export. Packing and
unpacking preserve signed payloads and require no signing key or GPU. Layer payloads are not grouped by a size threshold. Internal
logical-resource packages use envelope 6 with monolithic, numbered-shard, or
blob layouts. Legacy blob publication uses a fixed maximum of 64 GiB per file;
there is no configurable shard-size threshold.

`ignition export --no-weights` requires the exact stable and tuned weight
payloads in the cache. Initial state and execution resources may change with
the prepared capacities. Preparation and weight reads still occur. Successful
exports select their cache entry as the model's default. Recipes can declare
`export_repository` as the Hugging Face publishing target; prepared storage
uses the shared Flare cache. Multi-host export publishes the complete entry
in the coordinator's cache. See [package snapshots](cli-design.md#package-snapshots) for cache
management, conversion, and upload.

`ignition export --reuse-weights` uses the model's cached default package as
the weight source. It reuses that version's signed weight hashes and initializer
mappings, skips reading declared numeric weight storage during export, and
hardlinks the cached payloads into the resulting package without rehashing their
contents. Checkpoint loading
and model preparation still run. The checkpoint revision, TP size, architecture,
and declared weight layouts must match; missing or incompatible weights fail
export. Cached weight bytes are authoritative and are not compared with live
tensor values. Use ordinary export after changing weight values or packing.
Reuse includes Engram's file-backed embedding weights and scale planes, in
both shared and row-partitioned table layouts. Their geometry, quantization,
vocabulary map, and hashing parameters must match the cached version.
Graphs and initial state are always exported from the prepared model. See
[export flags](cli-design.md#31-ignition-command-line-flags) for the metadata requirements.

Each tensor-parallel deployment contains independently prepared rank metadata,
graphs, and kernels. Combining exported deployments can share or reconstruct
byte-identical weights; it cannot create an unexported TP size. Ignition creates
a per-user development signing key when `[publishing]` is absent. `flare run`
needs the corresponding publisher profile and first-use acceptance. See
[signing and development](cli-design.md#signing-and-local-development).
Unsigned rank staging is private to preparation and cannot load in Helios.

Flare loads the package files without Python. Helios is a Rust library;
`flare_heliopause` is a CPython extension imported by Ignition.

Recipes default to FP8 E4M3 KV storage, including learned draft-model caches. Qwen Flash retains BF16 selector keys;
GLM and DeepSeek retain their packed latent formats, including scale metadata and any BF16 rotary fields.
Recurrent state, compressor accumulators, and hidden activations retain their
model-specific precision. Cache writers and attention readers use matching
scales where required, and memory planning includes each cache's declared page
precision and partial-page snapshots.
Export a package to use these cache layouts; existing packages retain the
layouts captured when they were prepared.

## Environment setup

Set up the preparation environment and build the native CLI:

```sh
uv sync --locked --extra prepare --extra test
cargo build --release --manifest-path flare/Cargo.toml
```

Ignition applies three bundled preparation patches to the pinned PyTorch 2.14.1 and
Triton 3.8 preparation environment before importing preparation libraries in
`export`, `debug-serve`, `bench`, and `join`, including commands launched through `uv run`.
`ignition/patches/torch-inductor-triton-cache-filenames.patch` matches
Triton's 150-character physical cache filenames while retaining full CUDA symbol
names. `ignition/patches/torch-inductor-cooperative-static-launch.patch` preserves
cooperative vision reductions in portable static kernels through Triton's CUDA
launcher, including their cooperative launch attributes and frozen selections.
`ignition/patches/triton-platform-key-pointer-width.patch` derives Triton's helper
cache key from the interpreter's pointer width without launching `file`.
Nsight Systems injection can conflict with `file`'s seccomp sandbox and hang
startup ([NVIDIA known issues](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html#general-issues)).
The pointer-width key retains machine and operating-system identity and
invalidates compiled helper caches that used executable linkage metadata.
Patch application checks dependency versions and exact source context, accepts
already applied hunks, and replaces installation files atomically without
modifying uv's shared wheel cache. Each preparation host needs a writable,
dedicated Flare environment. Reinstalling Torch or Triton restores its distribution files;
the preparation command reapplies the patches.

`uv sync` installs dependencies without running the patches, and `uv build`
includes the patch files in Ignition's wheel and source distribution without
modifying its isolated build environment. For standalone Python tools and tests
that import Inductor or Triton before invoking Ignition preparation, apply the bundled
patches before starting their process:

```sh
uv run --locked --extra prepare python -m ignition.patches
```

For the platform Torch 2.9.0/Triton 3.5.0 environment, keep the PPU artifacts
and select their patch set explicitly before standalone tools or pytest:

```sh
python -m ignition.patches --operator-target ppu10
```

Use `ppu15` for that target. This edits only Triton's `runtime/build.py` helper
cache key and leaves installed Torch files unchanged. It does not rebuild
either package. `qualify.ppu_box` applies and checks this same fix before
importing operators, including when inspecting interfaces without device tests.

For Torch 2.9, framework compilation sets `triton.unique_kernel_names=False`:
actual symbols stay short (`triton_`), so static lookup and reload agree with
Triton's cache filenames without a Torch patch. Content hashes keep different
kernels separate; generated wrapper names retain operation descriptions. Raw
profiler symbol names are less informative. The option reaches compiler workers
and participates in forward-cache identity, so new preparation cache entries
are built on demand without deleting old ones. Exported runtime packages remain
loadable. An already applied Torch filename patch can stay in place, but new
PPU setup no longer applies or requires it.

`python scripts/setup_preparation.py` resolves the FlashInfer revision declared
in `pyproject.toml`, syncs preparation/test dependencies, rebuilds FlashInfer and
Heliopause, and applies the bundled patches before exiting. It updates `uv.lock`
when the declared dependency resolves differently.

Build dependencies include a C/C++ compiler, CMake, Ninja, pkg-config, Protobuf's
`protoc`, OpenSSL headers, Python headers, and liburing headers. On Ubuntu:

```sh
sudo apt-get install build-essential cmake ninja-build pkg-config protobuf-compiler libssl-dev python3-dev liburing-dev
```

Windows native builds also require Perl for the bundled static OpenSSL build,
alongside the C/C++ toolchain. The resulting binaries do not require OpenSSL DLLs.

The JPEG decoder builds its bundled libjpeg-turbo source. Preparation requires
the CUDA toolkit, including `nvcc` and CUPTI. Native build libraries discover
the system toolkit; `[paths] cuda` overrides that discovery with an explicit
toolkit directory.

The `prepare` extra installs `cuda-tile[tileiras]`, including the `tileiras`
compiler toolchain required by cuTile kernels during model preparation.
cuTile discovers its matching compiler tools in the Python environment
independently of the toolkit used for native builds.

The DeepSeek Rust crates share a pinned revision of
[`local-inference-lab/deepseek-recipe`](https://github.com/local-inference-lab/deepseek-recipe)
in `helios/Cargo.toml`. Its `flare` branch exposes normalized-message byte
boundaries, advisory checkpoint hints, and numeric V4.1 reasoning effort.
To update these crates, integrate upstream changes
on that branch, validate the encoder, and update all three dependency revisions
together. Refresh and commit `helios/Cargo.lock`, `flare/Cargo.lock`, and
`heliopause/Cargo.lock`; run the Helios frontend/protocol tests and build Flare
and Heliopause against the pinned revision. Compare rendered prompts and token
IDs with upstream when changing encoder behavior.

The preparation dependency pins a revision of `local-inference-lab/flashinfer`
in `pyproject.toml`; `uv.lock` records the same commit. To update FlashInfer,
change that revision and run `uv lock --upgrade-package flashinfer-python`,
then sync the environment.
For local FlashInfer development, install the sibling
checkout explicitly with `uv pip install --python .venv/bin/python --no-deps -e ../flashinfer`;
`uv sync --locked --extra prepare --extra test` restores the Git dependency.
Inspect that checkout and its local changes before modifying it. After changing
Rust code in `heliopause/`, or Helios code used by the extension, rebuild the
installed extension explicitly:

```sh
uv pip install --python .venv/bin/python --no-deps ./heliopause
```

## GPU selection

Inspect GPU occupancy with `nvidia-smi` before preparation or loading. Select
available GPUs explicitly on shared hosts. `--gpus 2` means one rank on GPU 2;
`--gpus 0,1` means two ranks on GPUs 0 and 1. With `CUDA_VISIBLE_DEVICES` set,
indices refer to the exposed device list. Each tensor-parallel size requires a
separately prepared package; a TP2 package cannot be loaded as TP1. Qwen TP1 and TP2 and
GLM TP4 have been exercised end to end; Qwen TP4 still requires model validation.

## Preparation resources

`--autotune-effort off|default|audit` controls b12x candidate filtering.
`off` prepares heuristic choices without tuning races or tuned-selection cache
lookups. `default` races the candidates that pass each kernel family's
performance exclusions. `audit` lifts those exclusions and races every
candidate satisfying correctness, resource, dominance, and explicit
configuration constraints; it exists to check that the default search prunes
no winners and is meant for occasional sampled runs, not routine tuning. The
audit search includes every default candidate. Kernel families retain their
declared tuning axes.

`--collective-tuning static|runtime` selects when collective choices become fixed.
`static` uses ordinary preparation-time autotuning and emits no collective startup
tuners. Every recipe defaults to `static`. `[preparation] collective_tuning`
overrides the recipe default; the command-line option overrides configuration.
`runtime` requires `--collective-ladder PATH` and uses the file's final joint
kernel geometry for eligible plain BF16 direct PCIe reductions. Preparation
compiles that geometry without racing alternatives. A captured tuner selects
among the supplied submode and active-CTA profiles when the package loads,
including with autotune effort `off`; geometry stays fixed and no search runs
during requests. Other collective paths retain their preparation policy. The
[GB202 PCIe TP4 ladder](../ignition/preparation/ladders/README.md) covers four
SM120 GPUs with out-of-place BF16 reductions from 16 bytes through 8 MiB:

```sh
--collective-tuning runtime \
--collective-ladder ignition/preparation/ladders/gb202_pcie_tp4.json
```

`--lock-gpu-clocks` locks each participating GPU to its maximum supported
graphics frequency for the b12x tuning batch and actual Inductor autotuning
trials during compiled-forward preparation. Frozen warmup, capture, and serving
run without these locks. It applies to serving ranks and
extra tuning replicas, and does nothing with `--autotune-effort off`. The GPU's
UUID selects the device under `CUDA_VISIBLE_DEVICES`. Run with permission to
control clocks: Ignition uses `sudo -n nvidia-smi` when not root. Unsupported
clock control or missing permission fails preparation. Locks reset to driver
defaults after tuning or a Python exception; preexisting manual clock locks
are not restored. Memory clocks and tuning caches are unchanged.

Without extra tuning replicas, tensor-parallel ranks on identical GPUs divide
each kernel race's candidates and adopt the fastest measured configuration.
This applies within one host as well as across hosts. Fixed collective kernels
still prepare on every participating rank.

Set `[preparation] autotune_effort = "audit"` for a persistent setting; an
explicit flag wins. The plan carries the setting to every preparation rank
and tuning replica. Effort separates cached selections and prepared packages;
identical kernel programs share compilation artifacts across efforts.

Tuning checkpoints each completed measurement batch in a SQLite database beside
the kernel selection cache. Tensor-parallel ranks save independently, including
completed local races waiting for winner exchange. Restarting the same workload
restores measured candidates and the carried winner, then measures unfinished
candidates against that winner. The progress display distinguishes restored,
newly measured, and rejected candidates. An interruption loses the unfinished
measurement batch on each rank; compilation artifacts remain separately cached.

Partial checkpoints require the same candidate program identities, effort,
tuning shard count, and measurement settings. A kernel or predicate change
invalidates the affected race's checkpoint. Completed selection caches remain
usable independently of the tuning shard count. Resource-dependent cooperative
launch rejections are retried after restart; fatal CUDA errors still require
restarting preparation in fresh processes.

`--compile-workers N` sets a positive CPU compiler process budget per host for
b12x kernel compilation and complete-forward compilation. Available RAM and
device memory can reduce it; local preparation ranks share the budget. Set
`[preparation] compile_workers = N` for a persistent default; the CLI flag
overrides that setting. Omitting both retains automatic sizing, including
shared-memory limits on DGX Spark.

On integrated GPUs, kernel tuning is divided into memory-bounded jobs. The
terminal dashboard shows the active job number and total job count, with a
separate live display for each job. Redirecting output records plain kernel
progress milestones instead of the terminal dashboard.

`--tuning-gpus LIST` adds local GPU replicas for benchmarking b12x candidates.
The list begins with `--gpus` and continues in complete TP groups: for example,
`--gpus 0 --tuning-gpus 0-3 --compile-workers 24` prepares a TP1 package with
four tuning GPUs. Corresponding GPUs in each replica must match in model and
architecture and have enough free memory for the model and preparation storage.
Helpers exit and release their resources after tuning; serving TP size and capture
devices stay fixed. Serving ranks then compile and capture their forwards.
Both GPU lists use `CUDA_VISIBLE_DEVICES` ordinals. Kernel compilation runs in
CPU workers, independently of the number of tuning GPUs.

Multi-host `--ranks` preparation retains its existing sharding and transport;
`--tuning-gpus` is local-only and cannot be combined with it. An explicit
`--compile-workers` budget applies separately on each host.

## Exporting runtime packages

Export publishes a directory entry in the shared model cache and prints its
path. In the examples below, replace `CACHE_ENTRY` with that path before running
`ignition convert`. Conversion requires an unused destination and does not append
a suffix. Do not delete an existing package merely to reuse its name. `MODEL` can also be a
supported repository ID or a local source-checkpoint directory. Ignition
downloads missing source weights, loads and tunes kernels, compiles forwards,
warms and captures execution variants, writes the package, and exits.
Compilation is enabled by default; `--no-compile` is for development.
The examples omit default settings. Configuration-file overrides still apply;
inspect `ignition export MODEL --help` and the printed preparation plan for the
resolved capacities. Select idle GPUs explicitly. Native serving listens on
`127.0.0.1:8000` unless configuration or `--host` / `--port` overrides it.
Add `--verbose` for per-phase detail or `--lock-gpu-clocks` to lock clocks during
autotuning.

### Re-exporting with cached weights

After an ordinary export, use `--reuse-weights` when rebuilding graphs with
the same checkpoint, TP size, architecture, and compatible weight layouts.
For a DeepSeek V4.1 TP4 export:

```sh
.venv/bin/ignition export deepseek-v4.1-flash --gpus 0,1,2,3 --reuse-weights
flare/target/release/flare run CACHE_ENTRY --gpus 0,1,2,3
```

Replace `CACHE_ENTRY` with the directory printed by this export. The command
uses the model's cached default version as its weight source and publishes
the resulting package as the default. Checkpoint loading, preparation, and
capture still run; compatible tuning and compilation caches remain reusable.
Use ordinary export after changing weight values or packing, or when no
compatible cached default exists. `--no-weights` instead hashes the live
weights and requires their exact payloads to be cached; it cannot be combined
with `--reuse-weights`.

## Nemotron 3.5 Bolt 74B-A9B 24GB

`nemotron-bolt-24gb` prepares
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` on one SM120/SM121 GPU.
The recipe includes the C-RADIO encoder for still images. Video is not prepared. LiLiCorr uses the matching
`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` checkpoint and defaults to
seven draft tokens with adaptive verification. Both checkpoints use the standard
Hugging Face hub cache (`~/.cache/huggingface/hub`), or `[paths].hub_cache` when configured.

This example limits preparation to eight concurrent requests and 64 target rows:

```sh
.venv/bin/ignition export nemotron-bolt-24gb --gpus 0 \
  --max-concurrency 8 --speculative-token-capacity 64
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/nemotron-bolt-images-lilicorr-c8.safetensors

./flare/target/release/flare run \
  /data/flare-runtime/nemotron-bolt-images-lilicorr-c8.safetensors \
  --gpus 0
```

Use `--no-mtp` during preparation to omit the draft model. Mamba-2 prefill
uses b12x's packed selective scan; decode and speculative verification use
FlashInfer state updates. Continuation rows retain BF16 recurrent state,
convolution history, partial FP8 KV pages, and LiLiCorr context and rollback
records. Prefix reuse and speculative acceptance preserve that complete state
through the shared Helios cache contract. Image features enter causal prefill
and may cross chunk boundaries.

Shared experts overlap routing and routed experts for graph capacities through
64 rows. Larger capacities execute those branches serially. LiLiCorr's separately
scaled NVFP4 gate and up projections also overlap. Prepare with
`--no-stream-overlap` to place those projections and expert branches on the main stream.
Routing excludes padded target rows using the packed batch's valid-token count.
The 64-row speculative capacity covers eight requests with seven draft tokens each.
For concurrency 32 with seven draft tokens per request, use
`--speculative-token-capacity 256`.

## Qwen3.8 27B

`qwen3.8-27b` prepares `nvidia/Qwen3.8-27B-NVFP4` for text generation on one GPU.
It uses the canonical Qwen3.5 dense decoder with tensor-scaled FP8 attention
projections and NVFP4 MLP and vocabulary weights. FP8 KV pages and Gated DeltaNet
continuations reside in device memory. The pinned BF16
`incoai/Qwen3.8-27B-DFlash2` draft consumes residual features from target layers
5, 19, 33, 47, and 61 before final normalization, shares the target embedding and
vocabulary head. The prepared noncausal proposal block contains the pending token
and the configured maximum draft count, supporting one through sixteen positions;
the checkpoint retains its declared eight-token block metadata. Speculation defaults
to seven draft tokens with adaptive packed verification capped at 128 physical
target rows and a 64-graph budget. Accepted GDN corrections commit into one
recurrent row per request. Five 2,048-token FP8 draft KV rings per continuation
track target consumption without lag and survive prefix snapshot/restore.
The recipe has no host-mapped tables or vision graphs. Prepare on the same GPU
architecture as the serving device:

```sh
.venv/bin/ignition export qwen3.8-27b --gpus 0
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/qwen38-27b-tp1.safetensors
```

`qwen3.8-27b-qad` selects the pinned
`local-inference-lab/Qwen3.8-27B-NVFP4-QAD` checkpoint. Its calibrated NVFP4 MLPs,
MXFP8 GDN projections, BF16 full-attention and vocabulary weights, and trained
FP32 normalization weights retain their stored precision by default.
`--online-mxfp8` converts eligible BF16 target attention/MLP projections and
DFlash2 context, attention, and MLP projections to MXFP8 while loading. Existing
packed weights, normalization, grouped-convolution parameters, and candidate
selector weights are unchanged. With a BF16 checkpoint vocabulary head,
`--draft-head-quantization nvfp4` or `mxfp8` creates a separate proposal head;
the target vocabulary head retains BF16 precision. For example:

```sh
.venv/bin/ignition bench qwen3.8-27b-qad --online-mxfp8 --draft-head-quantization nvfp4 --gpus 0
```

Use `--mtp-draft-tokens N` to select zero through fifteen prepared draft tokens.
Width-one rounds retain draft context so adaptive verification can resume wider
rounds. The proposal block, GDN correction records, and convolution rollback
storage derive from the configured maximum verification width. Packed target
capacity remains independent of the logical request-by-width sampling capacity.
Planning reserves 193,865,984 continuation bytes per row, the supported
sixteen-position maximum; narrower preparations allocate their exact smaller
continuation layout.

For eight concurrent requests with fifteen drafts and a 64-token packed target
budget:

```sh
.venv/bin/ignition export qwen3.8-27b-qad --gpus 0 --max-concurrency 8 \
  --prefill-chunk-size 4096 --mtp-draft-tokens 15 \
  --speculative-token-capacity 64 --speculative-graph-budget 64 \
  --online-mxfp8 --draft-head-quantization mxfp8
```

The recipe context capacity is 262144 tokens.
`--no-mtp` skips the draft download and omits its weights and state during preparation;
`flare run PACKAGE --mtp-draft-tokens 0` selects the package's ordinary graphs.

For complete GPU graph round latency without package export, use:

```sh
.venv/bin/ignition bench qwen3.8-27b --gpus 0 --output /tmp/qwen38-rounds.json
```

This prepares only the selected benchmark graph catalog. Defaults cover request
batches 1, 2, and 4, verification width eight, and consumed contexts 512, 2,048,
8,192, and 16,384 tokens. Timing uses real prefilled contexts and restores full
cache and speculative state before each sample, outside CUDA event timing.
The [benchmark CLI](cli-design.md#gpu-graph-latency-benchmark) defines workload
selection, repetitions, seeds, and the equal-weight latency score. Native
serving measurements still use an exported package with `flare run`.

To time a complete Inductor-compiled prefill graph at one exact chunk size:

```sh
.venv/bin/ignition bench qwen3.8-27b --gpus 0 --prefill \
  --prefill-chunk-size 4096 --contexts 0 --output /tmp/qwen38-prefill.json
```

`--contexts` specifies consumed tokens preceding the timed chunk. Admission,
prefix preparation, and state restoration remain outside timing. DFlash2
context assimilation is included by default; add `--no-mtp` for target-only
prefill. This mode prepares no decode or speculative round graphs.

## Qwen3.8 Flash

GGUF checkpoint inspection is available for development with
`uv sync --locked --extra gguf --extra test`. Call
`ignition.preparation.gguf.inspect_gguf_checkpoint(Path(...))` with a GGUF file
or any shard. It checks the shard set and tensor ranges and reports logical
shapes, original block encodings, and stored byte shapes without dequantization.
The external `gguf` package owns format parsing.

Development components can read these tensors with
`ignition.components.gguf_weights.GgufWeightLoader`. Quantized projections keep
their original blocks; row, expert and block-aligned column selections read only
the selected ranges. Gate/up fusion concatenates rows separately within each
expert. Plain weights retain their source dtype. The large
`per_layer_token_embd.weight` PLE table is excluded from resident loading.

`ignition.components.gguf_projection.PreparedGgufProjection` binds a dense
projection to the external [QUactlize NVIDIA compatibility operators](https://github.com/DrXuQian/QUactlize/tree/feat/nvidia-dequant-gemm).
Install that source package against the preparation environment's Torch and CUDA
toolkit. Query `projection_workspace_bytes(source, weight)` and supply fixed
byte storage before preparation warmup. Sequential projections share the maximum
required scratch; concurrent projections need disjoint storage. Weights remain
packed and the operator expands one matrix into scratch for each call. Its native
handle is rebound when Flare restores a compiled forward; keep the prepared
owner alive until graph work drains. This handle has QUactlize's ABI and cannot
be passed to b12x's projection overlap helpers.

Hosts whose containers prohibit `io_uring` can explicitly select
`read_mode='pread'` on the loader. This preparation-only transport retains native
O_DIRECT reads and checks source identities and destination overlap. Local GGUF
pread now uses POSIX reads and platform CUDA copies without importing b12x,
including when loading progress is enabled. Metadata-only Torch views describe
the selected ranges. Its pinned staging is bounded at 4 MiB plus 8 KiB for
alignment, with another 8 MiB only when converting stored BF16 to FP32. Other
checkpoint transports retain their existing readers. Read errors never switch
transport automatically.

Both local pread readers use `cuMemcpy2DAsync_v2` from the platform driver and
Torch's current context/stream. They do not initialize `cuda.bindings.runtime`:
on the PPU box that separate Runtime returned `cudaErrorInsufficientDriver` (35)
even for `cudaFree(0)`, while Torch could allocate device memory. The reported
driver and runtime library paths were already inside the PPU SDK. The driver
binding preserves exact byte pitches and the existing drain-before-reuse rule;
absence of Torch's current driver context is an error, not a new context request.
Host C-ABI checks pass against NVIDIA and PPU SDK headers. The user reports all
six box read cases below passing after this change; this qualifies the file/copy
boundary, not the native dense decoder or expert arithmetic.

Explicit PPU GGUF loading requires `pread` and prepared QUactlize helpers.
Target and shared-MTP dense expansion use the selected helpers on a private
preparation stream, retaining raw input and output until the work drains. The
NVIDIA debug decoder remains available on the original debug path.

GGUF preparation now connects packed routed experts, packed token lookup,
host-resident IQ4_NL PLE delivery, and the shared-Q8 MTP to the Qwen recipe.
Dense matrices are expanded once during preparation; expert weights remain
packed. The shared MTP uses the target embedding/head and ordinary attention
(compression ratio zero). Target QSA uses compression ratio four. Vocabulary
IDs, padding rows and BPE merges must match the pinned Flare frontend assets.

External GGUF error flags are consumed inside the complete captured forward,
after post-sampling MTP absorption. Repeated draft calls accumulate failures even
when the native ABI clears its flag on the next successful call. A failed round
uses the existing request error status/code 7, emits no tokens or logprobs, and
restores the logical consumed/generated counters. Already terminal requests,
inactive slots, stale generations and the other completion frame stay intact.
The current ABI reports batch-global errors, so all participating requests fail;
the native scheduler drains and retires their state without publishing a prefix.
This is request failure handling, not speculative rollback of a surviving request.
No host flag polling or additional graph launch is required. Actual external
lookup faults cover AOT, capture, two queued native frames and slot reuse.
An MTP accumulator also exposes its source flags to the graph boundary reset.
This handles sticky PPU errors without carrying a previous request's failure
into a newly admitted generation. Both clearing and sticky-bit semantics are
tested using actual CUDA lookup faults; this does not execute PPU kernels.

For a complete local checkpoint, select a directory containing `manifest.json`:

```json
{
  "schema": "qwen38-gguf/1",
  "main": "/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
  "mtp": "/models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
  "frontend": "/models/frontend",
  "read_mode": "pread"
}
```

On SM120, optionally add `"actlize_library": "/path/libactlize_la_serving.so"`
to select external actlizeLA GDN and QSA in `ignition export INPUT_DIRECTORY`.
Install its Python preparation package and build the native SM120 library first,
as described under [PPU port preparation](#ppu-port-preparation). The explicit
selection switches target and MTP KV pages and hybrid snapshots to BF16; MTP
keeps ordinary attention. The default without this field keeps b12x and FP8.
The plan reports the selected operators and budgets their actual page/snapshot
bytes while preserving the checkpoint alias and its model settings. The library
content and target enter the
checkpoint/preparation identities and compilation namespace. Changes between
planning and loading are rejected. Keep the input manifest and matching library
available for re-export; native serving uses only the exported package.

The NVIDIA implementation of QUactlize's PPU matmul records can also be selected
with `"quactlize_abi_library": "/path/_cuda_compat.so"`, independently or with
`actlize_library`. Install the `feat/nvidia-ppu-matmul-abi` package first. This
selection reaches target and shared-MTP experts and their own workspace queries.
It preserves the checkpoint's packed weight bytes and does not change the KV
precision chosen by the attention library. Library content and the record ABI
enter the reported plan, checkpoint revision and compilation namespace; changed
libraries are rejected before weights load. The normal preparation flow and
the bounded qualification command share the same explicit library loader.
This selector is NVIDIA-only; it does not select a physical PPU implementation.

The updated serving helpers are selected independently with
`"quactlize_helpers_library": "/path/libquactlize_serving.so"`. This uses
`quactlize.serving` router v2, rounded SwiGLU v2 and raw Q8_0/IQ4_NL row lookup
for target/shared-MTP experts and embeddings. The staged PLE lookup keeps its
existing readiness dependency. Router errors join the graph's existing
request-error guard. Library bytes enter planning and compilation identity;
preparation warms the helpers on an explicit stream and retains their owner
until all graphs drain. The partial fixture accepts the same selection through
`--quactlize-helpers-library`. Its device qualification is pending. The NVIDIA
serving compatibility DSO is for debugging and numerical checks.

This selection currently requires SM120 and the existing single-GPU GGUF
capacity; automatic device selection considers only GPUs matching the selected
library. The external GDN ABI does not support mixed prefill/decode partitions;
preparation rejects such a catalog. It preserves the existing graph families,
sampling and native scheduling. Configuration, snapshot and page-budget tests
cover the actual BF16 layouts, with 237 input/configuration/planning cases passing.
The runtime qualification below remains partial
because the complete main checkpoint is intentionally not downloaded.

The full recipe requires all main shards and validates their declared tensor
counts. It has not been qualified with the complete checkpoint. Do not rename
partial downloads to satisfy this validation.

The bounded development fixture uses complete blocks 44/47 from shard four,
all 512 experts in each selected block, and the complete shared MTP. Its token
embedding, output head and small PLE table are deterministic test data; generated
text is not a language-model quality check. It exports ordinary and speculative
graphs through the same sampling, commit, rollback and cache code:

```sh
.venv/bin/python -m qualify.qwen38_gguf \
  --target-shard /models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf \
  --metadata /models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mtp /models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --frontend /models/frontend --output /models/qualification
```

Only complete selected layers are accepted; this does not resume or load missing
shards. Export staging currently needs roughly three times the final package
size in free disk space. Remove an obsolete fixture package before re-exporting
when space is limited. Serve the package using the native Rust CLI, after the
preparation process exits. Use `--kv-cache-limit memory` to retain enough backing
for hybrid prefix snapshots in this fixture. A successfully exported package
alone does not qualify request behavior or absence of GPU idle intervals.

Native RTX qualification uses batch one, context 1024, prefill 64, RAM table
delivery and the exported package after Python exits. Eighteen fixed-MTP request
cases and 50 ordinary/teacher-prefix cases pass, including cancellation, drained
slot reuse, and rejection of invalid token IDs before admission. Within each fixed mode,
restoring a cached prefix preserves generated IDs and logprobs exactly. Adaptive
service exercises widths 1/2/4/6. The actual-weight GDN test additionally checks
width 1/6 and committed prefixes 0–6 against Flare's existing checkpoint reference,
retaining its FP32 state bound (rtol 1e-5, atol 1e-6) and gated-output checks.
Cross-variant target scores are not bitwise identical: reconstructed raw-logit
relative L2 is 0.00242 overall and at most 0.00742 per teacher-forced position.

The fixed-width-six Nsight run contains 102 complete speculative rounds, each
627 compute kernels under one graph launch. Median kernel span is 11.330 ms;
consecutive-round gaps are median 15.553 us, p95 16.385 us, maximum 16.641 us.
No GPU allocation, module load, recapture or out-of-graph compute kernel occurs
during execution. Each round copies 8640 bytes of PLE rows plus small request
metadata; expert weights remain resident. Kernel-to-kernel dependency gaps
still exist. Three adaptive traces contain isolated submission gaps of
132–368 us; two also contain external PLE readiness waits of 0.956/1.523 ms.
The 3483-round repeat has 3420 consecutive width-one pairs: median 14.912 us,
p95 16.096 us, maximum 132.031 us. Cgroup counters show no CPU throttling during
that repeat. OSRT and thread schedstat records do not establish the cause of
the tails; this container prohibits CPU context-switch profiling. The same
48-request workload without Nsight observes all 3531 readiness-counter increments
individually, maximum 3.712 us; its millisecond waits do not reproduce. These
partial-model measurements do not establish full-model throughput or literal
zero idle.

Further analysis separates late host submission from weight traffic. In the
original 367.513 us tail, a `cuEventQuery` invocation spans 2.481 ms. The next
`cuGraphLaunch` call starts 351.686 us after the preceding graph's final kernel. The
141.311/132.031 us tails also submit their next graphs late. This identifies
the delayed boundary, not whether the delay came from the driver, profiling,
interrupts or CPU scheduling. No weight upload explains these intervals.

A diagnostic run of the current QUactlize-record/GDN/QSA package covers 3311
adaptive rounds across 48 requests. Its 3072 consecutive width-one pairs have
15.328 us median, 16.064 us p95 and one 448.282 us maximum gap. The maximum
follows a 2.798 ms `cuEventQuery`; the next `cuGraphLaunch` starts 398.670 us
after the preceding graph's last kernel. A temporary host-only yield probe
records wall and thread-CPU clocks, but neither recorded long model-thread
yield overlaps that tail. The trace still has no hot GPU allocation, recapture,
module load or out-of-graph compute. Removing one possible delay source does
not follow from this trace alone.

An unprofiled control does observe a 2.941 ms model-thread yield with only
1.596 us of thread CPU time. To test whether removing voluntary yielding helps,
a diagnostic wrapper keeps or skips only `flare-model`'s `sched_yield` in
keep/skip/skip/keep order. Each trial starts a fresh ordinary-decode service,
uses the same 48 requests and emits 6144 tokens; all token IDs match exactly.
Intervals below use the server's token publication timestamps, excluding
prefill and client/network observation delays. They measure host publication,
not GPU gaps or MTP graph latency. These trials load no Nsight instrumentation.

| Trial | Model-thread yield | Interior duration, summed over requests | p99.9 publication interval | Maximum interval |
| --- | --- | --- | --- | --- |
| 0 | Keep | 13.2895 s | 2.4242 ms | 5.9633 ms |
| 1 | Skip | 13.2419 s | 2.3761 ms | 4.0216 ms |
| 2 | Skip | 13.2422 s | 2.3905 ms | 4.6568 ms |
| 3 | Keep | 13.2400 s | 2.3798 ms | 4.5990 ms |

Skipping yields has no consistent advantage over the final control. The
wrapper is invoked 63.2/65.6 million times when skipping versus 35.5/35.9 million
when keeping the yield. The existing native polling loop is retained. The diagnostic libraries are absent
from the restored service. CPU context-switch tracing remains unavailable in
this container, so the rare driver/host tails remain open; these measurements
do not establish literal zero idle or a full-model performance result.

Correlating graph launches with the sequential workload locates the tails below.
Repeat, request and speculative-round numbers are one-based; each repeat uses
the same eight requests. Request 6 has a seven-token prompt, seed 435 and a
128-token output budget; request 8 uses seed 437 and a 256-token budget.

| Trace | Repeat / request | Round boundary | Inter-round gap |
| --- | --- | --- | --- |
| Earlier adaptive package | 1 / 6 | 33 to 34 | 367.513 us |
| Earlier package with OSRT | 1 / 8 | 138 to 139 | 141.311 us |
| Earlier package with scheduling observations | 1 / 8 | 98 to 99 | 132.031 us |
| Current package with yield probe | 6 / 6 | 18 to 19 | 448.282 us |

The 448.282 us boundary connects two executions of the same width-one graph,
each containing 249 kernels. The preceding graph ends with
`triton_poi_fused_copy_storage_call_unsqueeze_view_7`; the following graph begins
with integer-fill kernels that reset operator error flags. Its two surrounding
boundaries take 15.328/15.904 us. The same request and round boundary in the other
five repeats takes 15.424-16.160 us. No width transition, different graph, weight
upload or extra compute kernel explains this outlier. Relative to the preceding
graph's last kernel, the long event query returns at +300.710 us, the next input
batch copy is submitted at +358.486 us, `cuGraphLaunch` starts at +398.670 us,
and the next graph's first kernel starts at +448.282 us.

The queried event is the oldest pending frame's `transferred` event in
`GraphExecutor::poll`. Submission records `produced` after the complete graph,
makes the output-copy stream wait for it, copies completion records and any
sidecars, then records `transferred`. The CPU may read and recycle that frame
only after the query succeeds. This query does not separate target decode from
MTP. Two frames may be queued, so a slow host call need not empty the GPU queue.

A separate API probe records 7,130,900 queries in a 48-request Nsight run.
One query's 2.6516 ms diagnostic window agrees with a 2.6498 ms CUDA API trace;
thread CPU time is 2.6512 ms, with no observed context switch and 60 minor page
faults in the surrounding counter window. Two queued graph executions cover
that interval, so it does not cause a large inter-round gap. That run instead
has 342.107/488.570 us gaps at repeat/request 2/8 and 5/2, both at rounds 117 to
118. Their first subsequent queries start 138.403/368.130 us after the preceding
GPU work finishes. Thus late polling and long query calls are distinct
observations; the event query is not the common cause of every tail.

An unprofiled API control and a separate unprofiled signal-sampling control
also observe occasional long host intervals. Instruction samples inside an
active query reach `libcuda.so.1`; other samples occur after the CUDA call, in
the probe's thread-clock system call. The probe brackets include that clock
read, so their entire wall durations cannot all be assigned to CUDA. These
diagnostics affect timing and do not replace uninstrumented service performance
measurements. All 144 request choices and usage records across the three runs
match the earlier workload for identical inputs. The diagnostic libraries are
removed from the restored service. The observations narrow the investigation
to host submission, driver execution and measurement/environment effects; they
do not establish a specific internal driver, profiler or OS defect.

Graph-internal gaps are tracked separately. In the yield-probe trace, the first
uses of the speculative graph variants have 493.881-633.976 us gaps before
GGUF row-lookup kernels. The intervening node is a four-byte error-flag memset
taking about 0.320 us, not a weight transfer. Functions and graph executables
were already loaded and uploaded at package startup. The later API-probe trace
has two graph-internal gaps above 50 us, at 67.871/95.935 us before a width-two
Q8 row lookup. These measurements do not yet identify the cause of those gaps
or establish literal zero idle within the graph.

The native replay tool loads this same exported package and compares isolated
rounds with queued execution. Four 128-token cases each execute 22 width-six
rounds. Tokens, logprobs and completion payloads match exactly; request generations
and submission fences are validated independently. Each case compares 165465088
bytes of committed live hybrid state and the declared KV ranges byte for byte.
The service/replay ratios over the same interior 21 rounds are 0.9959–0.9992,
including SSE serialization and loopback delivery. Differences this small are
measurement variation; all four cases pass the 5% native-overhead limit. This
qualifies the partial fixture at batch one and fixed width six, with RAM PLE.

After adding captured external-error handling, all 18 MTP and 50 ordinary
request cases retain the baseline tokens and logprobs exactly. Four paired
native replay cases retain identical completion records and SHA-256 fingerprints
for all 40 state/KV ranges (165465088 bytes per case). Their median GPU-round
cost increases by 0.52–0.53% with sticky-source reset included. The updated
service/replay ratios are 0.9953–1.0012; all four remain within the 5% overhead
limit. Hashing and state readback are outside the measured rounds. These comparisons use the same partial
checkpoint, graph capacities, fixed five drafts and RAM PLE configuration.
The updated 102-round Nsight trace has 668 kernels per complete width-six graph.
Across 94 consecutive pairs, inter-round gaps are median 15.697 us, p95
16.417 us and maximum 16.577 us. A separate adaptive trace exercises 573 rounds
at widths 1/2/4/6. Its 560 consecutive width-one pairs have median 15.392 us,
p95 16.160 us and maximum 108.259 us; this isolated tail remains unattributed.
Neither trace has execution-time GPU allocation, recapture, module loading or
out-of-graph compute. Internal dependency gaps remain; this does not assert
literal zero idle or qualify other capacities.

Run that comparison against a fresh service using `--mtp-draft-tokens 5
--no-mtp-adaptive-verification`. The native tool explicitly trusts only the
repository's fixture publisher:

```sh
.venv/bin/python -m qualify.qwen38_gguf_replay capture --output replay-corpus.json
# Stop the service before measuring the native replay tool.
cargo run --release --locked --manifest-path helios/Cargo.toml \
  --features server,runtime-package,test-tools --bin test_qwen38_gguf_replay_floor \
  -- /models/qualification/package replay-corpus.json replay-floor.json
python -m qualify.qwen38_gguf_replay compare --corpus replay-corpus.json \
  --replay replay-floor.json --output replay-comparison.json
```

Run request checks against an otherwise idle native service:

```sh
python -m qualify.qwen38_gguf_http --url http://127.0.0.1:18081 --output mtp-native.json
# Restart the same package with --mtp-draft-tokens 0, then:
python -m qualify.qwen38_gguf_http --url http://127.0.0.1:18081 \
  --output ordinary-native.json --teacher mtp-native.json
```

### PPU port preparation

The complete QUactlize wheel is available in the
[`0.2.0` packaging change at a4207d6](https://github.com/DrXuQian/QUactlize/tree/a4207d6b74571235f741ffc0ab7a6bc16e9d2a28).
Install the built wheel with `python -m pip install --no-deps /path/to/quactlize-0.2.0-py3-none-linux_x86_64.whl`.
When it is published in the configured PPU artifacts index, use
`python -m pip install --only-binary=quactlize quactlize==0.2.0`.
This artifact declares **ppu10**, built with SDK **2.2.1-184ee6**. It is not a
ppu15/ppu17 or NVIDIA binary; select a wheel built for the actual platform.
Installation and library discovery do not build device code or replace Torch.

For an explicit `operator_target: "ppu10"` or `"ppu15"`, the GGUF input manifest
may omit `quactlize_abi_library` and `quactlize_helpers_library`. Both roles then
resolve through `quactlize.get_library_info(target=...)` from the installed
package; explicit paths override their respective roles. Discovery validates the
package resources before device initialization. The reported preparation
identity includes both the public DSO and the complete package manifest, so a
private runtime or Python binding update invalidates the old identity. Resources
are checked again before native loading. A missing package, incompatible target
or changed resource is an error, with no source/system-library fallback.

The actual 0.2.0 wheel passes isolated installed imports/discovery and 40 native
host ABI queries with its matching SDK runtime. Framework selection, changed
private-resource identity and override checks pass on Torch2.9/Torch2.15. These
are host checks. The user now reports the ten-case packed GGUF device gate
passing on PPU; complete target/MTP and native package qualification remain open.

The complete actlizeLA serving wheel is available in
[`0.2.0` at 4ea342e](https://github.com/DrXuQian/actlizeLA/tree/4ea342e95c2b7fda14223f8b0997d308b94ab3f7).
Install the matching artifact with
`python -m pip install --no-deps /path/to/actlize_la-0.2.0-1ppu10version221184ee6-py3-none-linux_x86_64.whl`.
When that artifact is published in the configured PPU index, use
`python -m pip install --only-binary=actlize-la actlize-la==0.2.0`.
The checked release also targets **ppu10 / SDK 2.2.1-184ee6**; ppu15 requires
its own artifact. Installation does not build kernels or replace platform Torch.

For explicit PPU manifests, `actlize_library` is now optional:
`actlize_la.get_library_info(target=...)` resolves and verifies the complete
installed package. An explicit path still takes precedence and keeps the
existing `NativeLibrary(path, target=...)` contract. The sealed package manifest,
including Python bindings, enters preparation identity and is revalidated
before loading. Changed bindings invalidate prepared inputs even if the DSO
hash is unchanged. Missing resources and target mismatches fail without fallback.
The NVIDIA default continues to require explicit library selection.
Actual wheel discovery passes outside the checkout without importing Torch,
loading a DSO or probing a device. Native ABI inspection checks its 21 exports
and 11 structure sizes. The framework's 82 package/input/admission checks pass
on Torch2.9 and Torch2.15; these are host checks, not device qualification.

The canonical expert path requires QUactlize
[`feat/ppu-flash-next` at d90cb0d](https://github.com/DrXuQian/QUactlize/tree/d90cb0d880b9114768b98c8a69a0a5a6b07c3253)
or a compatible later revision, with matching Python declarations and a rebuilt
PPU DSO. This revision supplies Q5_1 canonical packing/packed native16 AUTO
entries and Q8_0 K640 geometry. Earlier serving-helper checks used
[`8519e9a`](https://github.com/DrXuQian/QUactlize/tree/8519e9a76830a513ab8ede806063a02590c710b7/quactlize/compat):
its 24 host tests pass, and `serving.cu` compiles with SDK
2.1.1 for `ppu_10`. This adds the missing rounding, routing and device-indexed
row contracts to the previous matmul ABI; the earlier 203 host ABI/reader/shape
checks were on 01810ca5. The separate SM120 debugging library builds with CUDA
12.8; its throughput is not a requirement of the PPU implementation.
The checked actlizeLA interface is
[`feat/flare-gdn-qsa-mtp` at b5696176](https://github.com/DrXuQian/actlizeLA/tree/b5696176ce66735afc6e72d079bc05018377523d/actlize_la/serving).
Its 91 host contract cases pass against a built SM120 library. Pointwise,
ShortConv, PLE hash and GDN sources compile for `ppu_10` and `ppu_15`.
These new interfaces have not run on a device here. The 42 SM120 operator
cases and package results below qualify the earlier cbcb4b9 implementation.

The separate `qualify.actlize_export` fixture now checks the GDN verify/commit
and Flare gated-normalization composition through AOT workers, cached artifacts
with fresh native handles, and the production fixed-VMM package exporter.
It releases the first prepared owner before testing the restored artifact.
Fifteen cases cover physical width six with record capacity eight, changing
widths 1/2/4/6, consumed prefixes 0..6, changed state slots, and inactive,
empty and null rows. Output and FP32 state satisfy Flare's existing checkpoint
reference bounds. After Python exits, 30 native replays compare 16024872 bytes
per case exactly, with no execution allocation or Python, Torch or actlizeLA
host library mappings. This qualifies that component boundary on SM120;
it does not qualify complete-model actlizeLA substitution or PPU execution.
The Qwen decoder explicitly disables intermediate prefill checkpoint exports,
which the external GDN ABI does not implement. That missing feature is not a
requirement of the current Qwen path; preserve the disabled setting when binding it.

The partial GGUF fixture also accepts `--actlize-library /path/libactlize_la_serving.so`
to select the checked SM120 GDN implementation inside complete serving graphs.
Install the external Python package with `pip install --no-deps /path/actlizeLA`;
its default Python-only installation does not build the native serving library.
Build that library separately for the explicit target, as described by the
operator package. This selects GDN alone by default; add `--actlize-qsa` to
include the external QSA transaction and BF16 cache storage, as described below.
Projections, ShortConv, gated normalization and MTP attention still use their
existing NVIDIA operators in this qualification.

The new preparation bindings using b5696176 also replace GDN ShortConv and
gated RMSNorm with actlizeLA. HC normalization, gate arithmetic and residual
combination use its pointwise helpers, with existing scratch supplying a
disjoint combine output before copying back to the residual. Selecting
`dense_blas` as well composes MTP feedback from whole-10240 normalization,
two BF16 projections and a BF16 broadcast sum. The hidden norm is not split
into four independent norms. These changes require fresh device qualification;
the historical NVIDIA measurements below do not establish their performance.

`PreparedGDN` binds the supplied library during preparation. Transient operands
reuse the shared scratch arena; each layer retains its correction records until
the existing speculative acceptance supplies consumed input counts. Positive
prefixes preserve ShortConv's encoded history for the next call. Zero consumption
restores a snapshot taken before that convolution. Separate device flags preserve
recurrence, snapshot, commit and restore errors through the outer request guard.
The graph catalog, native scheduling and sampling policy are unchanged.

Twelve actual-weight cases cover prefill capacities 1/17/64, in-place decode,
consumed prefixes 0..6, subsequent convolution retries, and two GDN owners
sharing activation/workspace storage while retaining independent records.
Existing numerical bounds are retained. The exported partial package passes
18 MTP and 50 ordinary/teacher-prefix HTTP cases, including exact prefix restore
within each mode. Its 18 MTP cases emit the same 239 tokens as the b12x package,
but scores differ: reconstructed synthetic-head raw-logit relative L2 is
0.003023 overall and at most 0.014921 per position. This is a measured comparison,
not a claim of bitwise arithmetic equivalence or full-model language quality.
Four native queued/isolated replay cases match tokens, logprobs and all 40
state/KV ranges exactly (165465088 bytes each). Service/replay ratios are
0.9881–1.0085, within the 5% overhead limit. The Rust service has no Python,
Torch or actlizeLA host library mapped. These measurements cover the GDN-only
substitution; PPU execution remains unqualified.

Nsight confirms the external recurrence/commit kernels inside the exported
graphs. The 102-round fixed-width-six trace has 679 kernels per round;
94 consecutive pairs have median 15.375 us, p95 16.351 us and maximum
16.479 us between rounds. Adaptive execution covers 573 rounds at widths
1/2/4/6; its 560 consecutive width-one pairs have median 15.391 us,
p95 16.128 us and maximum 16.320 us. Neither trace contains execution-time
GPU allocation, module loading, recapture or out-of-graph compute. These
bounded traces do not explain the isolated tails in the earlier packages.

With the checked `actlize_la.serving` package installed and its separate SM120
native library built, use an empty output directory:

```sh
.venv/bin/python -m qualify.actlize_export \
  --library /path/libactlize_la_serving.so --output /models/actlize-qualification
cargo run --release --locked --manifest-path helios/Cargo.toml \
  --features runtime-package,test-tools --bin test_actlize_runtime_package \
  -- /models/actlize-qualification
```

Automatic kernel discovery preserves C-ABI pointers through fixed virtual
addresses. Relocatable low-level packages require explicit parameter bindings;
automatic discovery alone does not reconstruct those bindings.

QSA's corresponding fixture is `qualify.actlize_qsa_export`, using the same
arguments and native replay binary. It binds actual Flare metadata and BF16
page storage. Seventeen cases compare against b12x's independent compression,
selection and sparse-attention references, including page/compression boundaries,
contexts 2048/2051, exact score ties, retries after consumed prefixes 0..6, and
inactive requests. Thirty-four native replays match 4753040 bytes per case,
including all cache history and selected positions. Execution allocates no CUDA
storage and loads no Python, Torch or operator host library. This is SM120
component qualification; it does not establish full PPU model execution.

`QSAState.allocate`, `HybridStateLayout.build` and `HybridStatePool.allocate`
accept an explicit `kv_dtype` for operator preparation. Select BF16 consistently
for the page pools and snapshot layout; checkpoint bindings reject a mismatch.
The existing NVIDIA recipe defaults remain FP8. Memory estimates use actual
tensor bytes: BF16 adds 786432 bytes per ordinary hybrid snapshot, or 917504
bytes with both MTP boundary pages. Twelve storage/maintenance tests pass,
including capacity 2/32, scattered rows and exact target/MTP page restoration.

For combined GDN/QSA qualification, add both
`--actlize-library /path/libactlize_la_serving.so --actlize-qsa` to the partial
GGUF export command. This binds the external append/compress/select/attention
transaction once, after the existing projections, normalization and rotary
operations. It does not also execute the old QSA KV writer. The selected
positions and attention output use existing scratch views; transient operator
workspace shares the preparation arena, while error flags remain persistent.
Page-pool replacement rebinds the native handle before compilation and capture.
Target QSA and ratio-zero MTP use BF16 pages with matching hybrid snapshots;
MTP retains ordinary paged attention. The ordinary recipe default remains FP8.

Three actual block-47 cases at capacities 1/6/64 compare the complete composed
attention against independent compression, selection and sparse-attention
references, including context 2051, scattered pages, page-pool replacement and
inactive rows. Three actual shared-Q8 MTP attention cases pass at capacities
1/7/128 with BF16 pages and changing live counts. Existing reference bounds
remain unchanged. The combined exported package passes all 18 MTP and 50
ordinary/teacher-prefix HTTP cases, including exact prefix restoration within
each mode. Four queued/isolated native replay cases compare all 40 state/KV
ranges exactly (207277056 bytes each). The additional 41811968 bytes follow
from BF16 KV storage and the matching snapshot. Service/replay ratios are
0.9957–1.0005, within the same 5% overhead limit. Routing the operator choice
through the shared MTP decoder factory preserves the previous partial package's
68 request choices and all four native state fingerprints exactly. This does not assert bitwise
equivalence to the FP8 package or full-model language quality.

The combined fixed-width-six trace has 102 rounds, 676 kernels per round and
11.597 ms median kernel span. Its 94 consecutive pairs have median 15.296 us,
p95 16.288 us and maximum 16.544 us between rounds. Adaptive execution covers
546 rounds at widths 1/2/4/6 (515/29/1/1 rounds respectively); the 507 consecutive
width-one pairs have median 15.296 us, p95 16.096 us and maximum 16.319 us.
Both traces contain the external GDN/QSA kernels inside exported graphs, with
no execution-time allocation, recapture, module loading or out-of-graph compute.
These bounded partial-model traces do not establish literal zero idle, explain
earlier isolated tails, or qualify the remaining PPU operators and target export.

| Operation | Required contract | Current PPU source and remaining work |
| --- | --- | --- |
| GGUF projections and MoE | Q5_1, Q8_0, Q2_K–Q6_K; H2560/I640; 512 experts, top10; BF16 operands/output, FP32 accumulation | QU d90cb0d canonical preparation is connected, including Q5_1 packed native16 and Q8_0 K640. Producer/consumer host queries pass. Raw bindings/replay passed on the explicit NVIDIA debug implementation; canonical device numerics and every production capacity still need PPU validation. |
| Expert workspace | Resident packed weights, bounded expansion, changing device IDs | Matmul v1 uses SIMT at M≤8 and grouped TC above eight, with expansion bounded by four experts × 512 output columns. Q5_1 packed AUTO uses direct SIMT/TC without expanded weights; indexed TC still needs routing scratch. Query each capacity and measure full forwards before changing expert chunk size. |
| Router | FP32 route weights, original tie rule, inactive rows | QU serving v2 selects raw logits with descending-ID ties, selected-logit softmax and a device live count. Framework binding and sticky error propagation are implemented; device qualification remains. |
| SwiGLU | BF16 SiLU followed by multiply and BF16 output | QU serving v2 explicitly supports `round_silu=True`. Shared/routed GGUF experts select it with the helper library. Earlier v1 arithmetic remains unchanged and is not used by this binding. |
| Token embedding / PLE | Device-indexed Q8_0 rows of width2560 and IQ4_NL rows of width160; inactive IDs and changing live counts | QU serving row lookup v1 accepts device int64 IDs and an int32 live count, retaining raw GGUF bytes. Embedding and staged PLE binding are implemented; qualify actual graphs on PPU. |
| GDN | FP32 state `[slot,Hv,V,K]`, Hk16/Hv48, compact per-layer records, consumed-input commit count | actlizeLA provides decode/verify/prefill/commit. Preserve zero-prefix convolution rollback, device errors and each layer's retained records; bind through complete compiled forwards. |
| QSA | Ratio4, 24 query/2 KV heads, D256, selected capacity2051, page64 | actlizeLA supplies append/compression/selection/attention with BF16 KV. Combined GDN/QSA partial GGUF serving and snapshot/restore are qualified on SM120; complete PPU recipe binding remains. Do not retain the old writer as well. |
| Shared MTP attention | Ratio0, paged causal GQA, D256, page64, changing device lengths | The user reports six FA3 API graph tests passing on PPU at widths1/7/128. Explicit FA3 selection now connects ordinary MTP attention; cache-only absorption uses a separate BF16 layout copy. The prepared copy/attention and full-forward compiler tests await device execution. |
| Dense BF16 / HC projections | Preserve actual dimensions, strided outputs and FP32 accumulation | Explicit `dense_blas` selection binds cuBLAS or acBLAS GemmEx with caller-owned scratch. CPU ABI and layout tests pass; compiled device capture and native export with this selection remain unqualified. |
| RMSNorm / HC / ShortConv / MTP feedback | Exact dtype/rounding, inactive rows and complete mutable history | actlizeLA bindings cover GDN convolution/norm, four-stream HC and full-width MTP feedback. External scratch survives tuning/trim; raw convolution slot0 uses its new valid-slot history contract. Device composition tests remain unrun locally. |
| PLE hash and gating | Read-only committed history, int64 hashes, dilation3, BF16 gate boundaries | External binding preserves staged IQ4_NL readiness and aliases only expired norm intermediates. Geometry/export host tests pass; four composed graph cases await a device. |
| RoPE / sampling / state helpers | Device lengths, request-local RNG and graph-safe mutable metadata | Qualify the PPU FlashInfer interfaces and target compilation. Installing FlashInfer alone does not prove its caller-owned sampling ABI. |
| Native graph export | Capture and export on the deployment platform; replay after Python exits | SDK2.1.1 declares graph/VMM/event APIs. Qualify the CUDA facade, module-image recorder, PPU ELF globals/parameters and relocations. Current export uses CUPTI and CUDA module metadata; API names alone do not prove this boundary. |

GGUF input manifests can select ordinary dense projections explicitly:

```json
"dense_blas": {
  "library": "/path/to/libacblas.so",
  "api": "acblas",
  "workspace_bytes": 4194304
}
```

On NVIDIA, select the matching `libcublas.so` and `"api": "cublas"`. The
library path is relative to the input directory unless absolute. The library
content, API and scratch size are part of preparation identity; scratch is
budgeted once. No pip installation or device-capability guess chooses this
library. Keep the platform's Torch and FlashInfer packages.

`ignition.components.blas_projection` binds the C API directly. It supports
unbiased BF16-by-BF16 projections with BF16 or FP32 output, and FP32-by-FP32
projections with FP32 output. Inputs and outputs can have padded row strides;
columns must be contiguous and write spans disjoint. FP32 accumulation is
requested without reduced-precision reduction or TF32 conversion. Preparation
owns one handle and workspace for serialized dense projections; concurrent
branches require separate resources. Output and workspace mutations are explicit
to compilation. Complete forwards warm the calls before capture, and serving
replays captured kernels without calling the Python or BLAS host API.

The ordinary projection dispatcher and the Qwen cache/configuration modules
(`attention_state`, `cache_layout`, `state_layout`, `config`) can be imported
without b12x. Cache, MTP rollback and snapshot layouts retain their existing
byte geometry. The explicit external path now uses FA3 for ordinary MTP
attention and a separate preparation session. PLE selects the new path when both
external helper libraries are present; a package-owned mapped mailbox avoids
its legacy b12x allocator.
The platform FlashInfer build must also supply
Flare's caller-owned top-k/top-p workspace interfaces and request-local chain
sampling RNG contract; a successful `import flashinfer` does not establish them.
The metadata serializer and guard pickler now identify loaded resource types
without importing unused b12x, CuTe or peer-storage modules. QSA/MTP scratch and
GGUF weight-class imports also work without those packages. The original
b12x model path retains its existing preparation and tuning lifecycle.

Run the host checks without loading a device library:

```sh
python -m pytest -q tests/components/test_blas_projection.py \
  tests/preparation/test_blas_inputs.py tests/preparation/test_qwen38_cache_layout.py \
  tests/preparation/test_qwen38_gguf_inputs.py tests/recipes/test_qwen38_prefill_config.py
```

The C fixture in these checks implements only host-side ABI arithmetic, not
cuBLAS/acBLAS device execution. With an assigned device and matching Torch,
run the separate graph checks, including complete-forward compilation, changed
inputs, padded outputs and explicit workspace rebinding:

```sh
FLARE_TEST_BLAS_LIBRARY=/path/to/libacblas.so FLARE_TEST_BLAS_API=acblas \
  python -m pytest -q tests/components/test_blas_projection_cuda.py
```

For the first PPU box qualification, use the platform's Python 3.12 environment
with its Torch and FlashInfer artifacts. Install only the small missing test
dependencies (`pytest`, `numpy`, `gguf==0.19.0`, `protobuf>=7.35.1,<8`) and this
repository with `pip install --no-deps -e .`. Do not use the NVIDIA `prepare`
extra for this check. Install the complete QUactlize and actlizeLA wheels
described above. With matching ppu10 libraries, the installed-wheel check is:

```sh
cd /sim/eec/shared/junfu.qx/spark-local-inference
python -m qualify.ppu_box \
  --actlize-target ppu10 \
  --blas-library /usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib/libcublas.so --blas-api cublas \
  --run-device-tests --output /path/to/new-ppu-test-directory
```

Select the actual matching target; the current 0.2.0 artifact supports ppu10.
Omit `--run-device-tests`
and `--output` to inspect interfaces first. The command checks real exported
symbols/descriptor sizes, valid Qwen matmul geometry, sampling workspace and
`row_independent_rng` support. It then runs the framework's BLAS/GGUF/helper
graph tests in separate processes. It also exercises the actual external GDN sequence,
HC, full-width MTP feedback and PLE bindings. Skips are failures of device qualification.
Results and logs stay in the new output directory; no model is downloaded.
For development, explicitly supply both `--quactlize-source` and
`--quactlize-library` to use a checkout and include its upstream serving suite.
Likewise, `--actlize-source` together with `--actlize-library` adds actlizeLA's
upstream numerical, convolution/transaction and captured graph suites.
An optional DSO override alone uses installed Python declarations. The normal
wheel check reports unavailable upstream source suites as NOT RUN; all
framework numerical/graph gates remain enabled. It does not claim the
upstream source suites passed.

To run only the next fourteen actlizeLA framework cases with installed wheels,
use the following on the matching ppu10 box. No operator source path or DSO
variable is needed; an existing explicit DSO variable remains an override:

```sh
FLARE_RUN_CUDA_TESTS=1 FLARE_ACTLIZE_TARGET=ppu10 \
FLARE_TEST_BLAS_LIBRARY=/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib/libcublas.so \
FLARE_TEST_BLAS_API=cublas python -m pytest -q \
  tests/components/test_actlize_gdn_sequence_cuda.py \
  tests/components/test_actlize_mixing_cuda.py \
  tests/components/test_actlize_ple_cuda.py --tb=short
```

Expect fourteen executed cases. These check consumed-prefix GDN/convolution
rollback including raw slot0, HC, full-width MTP feedback and staged IQ4_NL PLE
graph replay. The user reports all fourteen passing on PPU in 8.67s. They do
not cover complete speculative rounds, native PLE I/O latency, export or Rust serving.

The next gate checks QSA and sampling without requiring the native Rust worker
or any b12x/cuTile package. From the updated framework directory, run:

```sh
FLARE_RUN_CUDA_TESTS=1 FLARE_ACTLIZE_TARGET=ppu10 python -m pytest -q \
  tests/components/test_actlize_qsa_cuda.py \
  tests/components/test_external_sampling_cuda.py --tb=short
```

Expect eight executed cases. QSA checks widths 1/7/128, real BF16 page layouts,
cache boundaries, tied selection, consumed-prefix recovery, inactive state and
compiled disk reload with fresh handles. Its CPU reference keeps the original
Flare numerical bounds and matches that reference exactly in host comparison.
Sampling composes actual Flare draft proposals, FlashInfer filtering and chain
verification in each graph, using vocabulary 248320 and widths 1/7. It checks
shared scratch across batch 1/4, every rejection position, stale/inactive rows,
penalties, top-k/top-p/min-p and per-request RNG under repacking. Both suites
check replay allocation stability and have direct/catalog variants. Initial
catalog compilation takes longer than subsequent replay.
Torch2.9's skipped `torch.library.wrap_triton` call in probability normalization
now goes through the existing Dynamo-aware launch helper; four host tracing
cases pass on Torch2.9 and Torch2.15. After the FP32 classification correction
and bundled Inductor cache filename patch, the user reports both sampling
cases passing on PPU (204.13s including preparation) and all six QSA cases
passing (98.58s). The later short-symbol workaround passes offline compilation,
static construction and cold reload on unpatched Torch. The user also reports
seven passes, one deselected in 292.58s with that policy: all six QSA cases plus
sampling's catalog case. `-k catalog` matches the QSA function name as well as
its parameter values, so the earlier four-selected-case expectation was incorrect.
These are component graph gates, not complete target/MTP state commit,
native export/Rust replay or whole-round bubble measurements.

The GGUF helper test creates tiny Q8_0/IQ4_NL files and replays changing token
IDs, route counts, poisoned inactive rows and sticky repeated-call errors in
one graph. Its compilation test checks actual Flare storage lowering.
The PLE test stages rows from a small IQ4_NL file before replay, then checks
hash publication, lookup, projections, norm/gating, dilation3 convolution and
history in one graph. It covers changing packed lengths, raw slot0, inactive
rows and immutable committed token history. It does not time native host I/O.
The separate `attention-api` suite checks the installed platform FA3 wheel's
native `flash_attn_3.fwd` interface with page64, D256, 24/2 heads, widths
1/7/128, packed inactive rows and changing device lengths. It checks both
ordinary reads and KV append, exact cache mutation, caller-provided output
and allocation stability during graph replay. The wheel must enable paged KV,
append, varlen and D256. The user reports all six cases passing on the installed
PPU wheel. This is operator API evidence; it does not establish full-model execution.

Select that installed extension explicitly with
`"flash_attention3_library": "/path/to/site-packages/flash_attn_3/_C.so"` in the
GGUF input manifest, using its actual filename (which may include a Python ABI
suffix). For the partial-layer qualifier the corresponding option is
`--flash-attention3-library PATH`. Preparation checks that the imported extension
matches this path, binds its content hash into the plan, and budgets BF16 cache
pages and snapshots. FA3 reads the pages after a separate unquantized KV copy;
cache-only MTP absorption performs that copy without attention. Cache-write bounds
errors participate in the existing round error accumulator. No b12x/cuTile import
is required by these prepared components. FA3 selects its own split schedule;
temporary attention buffers are allocated during warmup/capture in the graph pool.
The new `attention-prepared` suite checks actual copy/attention replay at widths
1/7/128, both directly captured and through Flare's complete-forward compiler,
with changed lengths, inactive rows, exact cache writes and allocation stability.
A cache-only case checks full-width int64 bounds and error retention across two
calls. The user reports all seven cases passing on PPU, including the three
compiled forwards. That box uses Torch2.9/Triton3.5: short raw launch
grids are padded to three dimensions, and the optional make_fx diagnostic metadata
flag is passed only where supported. The standalone qualification compiler now
uses Torch2.9's real `CompiledArtifact.save/load` protocol with device-bound cache
identity. CPU artifact round trips preserve aliases, mutable outputs and fresh
storage-call bindings.

The complete catalog now has a Torch2.9 adapter using bundled AOT entries and
serialized Dynamo guards. Host validation covers worker metadata, deferred and
real storage, runtime handle rebinding, exact dispatch keys, and rejection of
changed shapes, scalar conditions and storage aliases. The saved catalog retains
the selected static Triton config and rejects unfinished tuning. Prepared Triton
launches use raw JIT calls during Dynamo tracing; Torch2.9 skips `wrap_triton`
there. Dispatcher-based `make_fx` tracing keeps the wrapped call. The user reports
all four catalog cases passing on the PPU box (4 passed, 7 deselected). The
following command reruns that gate without weights or QUactlize/actlizeLA DSOs:

```sh
FLARE_RUN_CUDA_TESTS=1 python -m pytest -q \
  tests/preparation/test_external_aot_cuda.py \
  tests/components/test_flash_attn3_prepared_cuda.py -k catalog --tb=short
```

This exercises a metadata compiler worker, disk reload without compilation,
frozen reduction kernels, fresh opaque bindings, and retained graph capture.
The three additional FA3 `catalog` cases cover widths 1/7/128 through that same
worker/cache path. Replay changes device inputs and checks outputs, cache writes
and allocation stability. These four passes extend the seven previously reported
FA3 component cases. Full-model preparation, PPU package export and Rust replay still
need separate qualification.

Selecting all five external libraries (QUactlize matmul, QUactlize helpers,
actlizeLA, dense BLAS and FA3) now uses `ExternalPreparationSession` for the text
bootstrap and partial-layer qualifier. It finalizes the complete scratch inventory
before compilation, retains native owners through capture, and drains the device
before release. ForwardCache still compiles and freezes complete forwards. The
legacy session remains in use for the original b12x model path. Missing external
libraries on an explicit PPU target are rejected before model preparation.
This removes the session's b12x/cuTile requirement; full-model hardware planning,
external operator numerics and native export still require qualification.
The ordinary `ignition export`/`debug-serve` entry now selects platform CUDA
discovery in a child process for a complete explicit PPU operator set. This
preserves the caller's visibility mask without initializing the parent's context
before final selection. Planning and weight admission retain single-device and
memory checks. CLI patch setup verifies the platform Torch2.9.0/Triton3.5.0
versions, accepting vendor suffixes, and applies the Triton helper-cache fix.
Keep those platform wheels; short compiled symbols avoid changing Torch files.
Host checks cover admission, visibility and patch application.
GGUF pread and dense decoding are now connected without b12x on the external
path. Operator numerics, device discovery and native export require box validation;
full PPU export remains unqualified.

The following device gate creates only tiny synthetic GGUF files and needs no
model download or external operator DSO. It tests exact bytes, expert gate/up
joins, selected blocks, BF16 conversion, source changes, destination overlap and
bounded transfers crossing a 4 MiB boundary:

```sh
FLARE_RUN_CUDA_TESTS=1 python -m pytest -q \
  tests/preparation/test_gguf_weights_cuda.py \
  -k 'native_reads or selected_blocks or changed_file or pread_' --tb=short
```

With `FLARE_QUACTLIZE_HELPERS_LIBRARY` set to the selected PPU DSO, also run
`tests/components/test_quactlize_helpers_cuda.py -k dense_preparation` through
pytest. Its four cases compare selected Q8_0/IQ4_NL dense ranges against
independent GGUF decoding in FP16 and BF16, with the NVIDIA debug module disabled.
The user reports the six read cases above passing on PPU and the four native
dense-decoder cases passing separately (4 passed, 10 deselected in 4.09s).
The decoder gate checks exact Q8_0/IQ4_NL selected ranges in FP16 and BF16.

Routed expert weights now use the selected QUactlize producer during loading.
Each target or shared-MTP gate/up group is read with the existing per-expert
concatenation, packed on a preparation stream, then drained before its raw
device copy is released. Down projection loading follows the same sequence.
The producer must report byte-neutral low/high/units planes and a complete
arrangement descriptor; the consumer queries that exact descriptor. Q8_0 has
no high plane. Only one projection needs both raw and packed storage at once.
Embedding/PLE lookup retains original GGUF blocks; dense projections retain
their once-decoded FP16/BF16 storage.

Q5_1 selects the explicit packed native16 AUTO entry: its query must report a
direct SIMT/TC route without expanded weights or grouped activation/output
buffers. Q8_0 and K formats use canonical storage through matmul v1; canonical
storage does not imply a direct packed TC kernel. Existing expert execution
chunks remain at most eight rows. No throughput improvement is claimed before
complete-forward measurement. The selected DSO hash and layout policy enter
preparation identity. Compilation metadata carries the plane tensors and
descriptor, while native export declares only those compressed planes as
weights. Serving never performs this packing.

The separate `gguf-packed` suite creates small GGUF expert files, without a
model download. After installing the complete wheel, run from the Flare checkout:

```sh
FLARE_RUN_CUDA_TESTS=1 python -m pytest -q \
  tests/preparation/test_gguf_packing_cuda.py --tb=short
```

The test discovers the installed wheel using its declared default target.
`FLARE_TEST_PPU_MATMUL_LIBRARY` remains an optional test-only DSO override.
Expect ten executed cases; skipped cases do not qualify a device.

Its ten cases exercise file read, GPU packing, raw-copy release and prepared
projection replay for Q5_1/Q8_0/Q4_K/Q5_K at widths 1/9. Q5_1 additionally checks
AOT worker compilation, disk-cache reload with a fresh native handle and no
recompilation. Routes change across duplicate, inactive and invalid IDs, with
poisoned inactive input, sticky error checks and allocation-stable replay.
Independent GGUF decoding supplies the native16 numerical reference. Host ABI,
metadata and fake-export checks pass on Torch2.9/Torch2.15. The user reports all
ten device cases passing on PPU. They are not complete target/MTP or native
package gates.

The next `gguf-expert-chain` suite composes router, gate/up, rounded SwiGLU,
down and weighted reduction for target Q4_K/Q5_1 and MTP Q8_0 experts inside
one captured graph per capacity. It keeps 512 experts, top10 and eight-row
chunks, with synthetic H256/I128 files totaling about 81 MiB. It alternates
widths 1/7/17 on shared scratch, checks tied/changing routes, poisoned inactive
inputs, invalid counts, sticky router failures and output bounds, then reloads
the compiled catalog with fresh native handles and compilation disabled.
This checks expert composition only; full-width projection geometry is covered
by the preceding gate. Attention, shared experts, sampling and MTP acceptance/
rollback are separate requirements. The user reports both cases passing on PPU,
together with the fourteen BLAS cases below (16 passed in 51.17s).

The user provided the platform `libcublas.so` below. Run the expert chain and
the existing fourteen BLAS graph checks with its `cublas` API:

```sh
FLARE_RUN_CUDA_TESTS=1 \
FLARE_TEST_BLAS_LIBRARY=/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib/libcublas.so \
FLARE_TEST_BLAS_API=cublas python -m pytest -q \
  tests/preparation/test_gguf_expert_chain_cuda.py \
  tests/components/test_blas_projection_cuda.py --tb=short
```

Expect sixteen executed cases. The direct and catalog variants are diagnostic
checks, not throughput measurements; neither requires a model download.

The user reports the following gate passing on PPU. It exercises this session through `capture_forwards`, with
alternating one-row and seven-row graphs sharing scratch, strided state reads,
active/inactive writes, exact persistent state and allocation-stable replay:

```sh
FLARE_RUN_CUDA_TESTS=1 python -m pytest -q \
  tests/preparation/test_external_session_cuda.py --tb=short
```

It needs no model download or external operator DSOs. It validates framework
state transfer and catalog orchestration, not a complete target/MTP round.
For partial-layer qualification, pass `--operator-target ppu10` or `ppu15`,
`--actlize-qsa`, and the BLAS/FA3 paths to `python -m qualify.qwen38_gguf`.
QUactlize matmul/helpers and actlizeLA use installed wheels unless paths are supplied.
The target must match the installed platform and library builds.

The manifest also accepts `"operator_target": "ppu10"` or `"ppu15"` for explicit
QUactlize/actlizeLA library selection. The default remains the NVIDIA debug target;
its QUactlize loader binds the current `quactlize_cuda_matmul_*_v1` C ABI directly.
Target and library content enter preparation identity. Selecting the external
session does not qualify full-model hardware planning or PPU export.
The inspection also checks the platform Torch APIs used by Flare's AOT
serialization, guard pickler and retained CUDA graphs, and round-trips tensor
metadata with aliases. Platform API compatibility must be established before
attempting complete compilation/export; this inspection does not compile or
replay a model.

Passing this command qualifies those components, not complete PPU model
serving. Full b12x-free recipe composition, composed MTP device qualification,
target native export/Rust replay and whole-round bubble measurements remain
separate requirements.

For the existing partial-layer NVIDIA qualification, append
`--dense-blas-library /path/to/libcublas.so --dense-blas-api cublas`
to `python -m qualify.qwen38_gguf` with its usual shard, MTP and frontend inputs.
`--dense-blas-workspace-bytes` defaults to 4 MiB in this fixture. This uses the
same selection in target and shared-MTP graph construction and does not require
the complete model. This new BLAS selection has not yet been exercised through
that export fixture; the fixture itself still needs the other NVIDIA operators.

`quactlize_ppu_matmul_run_v1` has sticky device error bits. The adapter must
clear and consume them at the intended graph boundary, retain all buffers and
handles, and propagate failures through existing request state. Never read
routing decisions back to the host. actlizeLA's supported build targets are
`ppu10`, `ppu15` and `cuda_sm120`; a PPU1.7 target is a separate qualification.

The current NVIDIA and new PPU matmul C ABIs require field-by-field bindings.
Their problem records are 52 and 88 bytes, and their sizes records are 80 and
120 bytes. Both buffer records are 152 bytes, but input/raw/ID/output byte-count
fields have different offsets. Reinterpreting one as the other is invalid;
the GGUF weight bytes remain unchanged. Prepared PPU custom operations must
also preserve the explicit element strides and query the selected library's
scratch requirements instead of reusing NVIDIA workspace assumptions.

`ignition.components.ppu_gguf_projection` now provides that raw-GGUF preparation
bridge using `quactlize.compat.native.Library` explicitly. It queries each shape
through the PPU ABI, represents padded byte spans correctly, and exposes output,
workspace and sticky-error mutations to compilation. Its owning rank retains
the native handle; compilation workers receive only tensor metadata and the
integer binding. The native `prepare` executes and synchronizes a warmup, so the
bridge supplies private initialized input and route IDs before any graph is
captured. The component exposes its sticky-error storage for the existing
complete-round error guard. An explicitly supplied projection library now reaches
target and shared-MTP expert preparation, including each scratch query. Normal
recipe inputs do not yet select the PPU implementation; target qualification
remains required. There is no automatic fallback or framework-owned matmul kernel.

Thirty-one checks pass against the actual QUactlize host contract and Torch
fake export: Qwen expert/dense shapes, SIMT/TC capacity selection, padded strides,
field-wise buffer extents, write alias rejection and all three mutable operands.
These host checks do not execute PPU kernels or prove compiled handle rebinding,
target export or device numerical behavior. Four device cases are supplied for native preparation and captured
replay with different addresses, duplicate/invalid/inactive IDs and sticky-error
reset, using the existing GGUF projection numerical bound; they remain unrun on PPU.
With the matching PPU Torch/QUactlize installation and assigned device, run:

```sh
FLARE_TEST_QUACTLIZE_PPU_SOURCE=/path/QUactlize \
FLARE_TEST_PPU_SDK=/path/ppu-sdk \
pytest tests/components/test_ppu_gguf_projection.py

FLARE_RUN_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=0 \
FLARE_TEST_PPU_MATMUL_LIBRARY=/path/libncp_quactlize.so \
pytest tests/components/test_ppu_gguf_projection_cuda.py
```

The separate QUactlize
[`feat/nvidia-ppu-matmul-abi` branch](https://github.com/DrXuQian/QUactlize/tree/feat/nvidia-ppu-matmul-abi)
provides `CudaPpuMatmulLibrary`: CUDA-named entry points consuming the exact
88/120/152-byte PPU records, with field-wise conversion to NVIDIA execution.
The existing NVIDIA API remains available. Both `setup.py` and `pyproject.toml`
package the shared Python records and public headers; an isolated wheel install
imports the new interface and queries it successfully. Its scratch query accounts
for selected-slot dequantization and cuBLAS workspace. This does not model PPU
grouped-TC memory or performance, and does not change the GGUF bytes.

On SM120, 132 QUactlize CUDA cases and four Flare capture cases pass, including
padded strides, changing addresses, repeated/invalid IDs and sticky errors. The
`qualify.quactlize_abi_export` fixture loads all 512 actual block-44 Q5_1 down
experts (600 MiB raw), with capacities 1/2/4/6/9/64 and padded input/ID/output
rows. Eighteen cases retain the independent GGUF-to-BF16/CPU-FP64 bound after
AOT cache reuse with fresh native handles. Thirty-six standalone Rust replays
compare 6571524 bytes each exactly, with no replay allocation or Python, Torch
or operator host library loaded. This validates the records and framework
boundary on NVIDIA; physical PPU execution remains unrun.

To exercise the same records in the actual partial target plus shared-MTP
package, pass `--quactlize-abi-library /path/_cuda_compat.so` to
`qualify.qwen38_gguf`, with that QUactlize package installed. It is an explicit
NVIDIA qualification option and composes with `--actlize-library` and
`--actlize-qsa`. The existing NVIDIA expert chunk limit of eight remains;
PPU M<=8 selects SIMT, so grouped-TC prefill needs its own complete-forward
measurement before choosing a target chunk size.

Normal GGUF input selection, planning, frontend and captured projection checks
pass 181 focused cases, including four actual GPU cases through the manifest
loader. A broader declaration/cleanup run has five failures also reproduced on
the unmodified preceding commit (three Kimi declaration cases and two cleanup
fixture cases), with 218 passing and 11 skipped; it is not reported as green.
No complete-checkpoint export is claimed by these input tests.

That combined package now preserves all 68 native HTTP choices and four native
replay cases' complete 207277056-byte state fingerprints exactly against the
preceding external-GDN/QSA package. Service/replay ratios are 0.9720–0.9757;
both include the same completion boundary and stay within the 5% overhead
limit. The fixed trace has 102 rounds, 676 kernels per round and 12.248 ms
median kernel span. Its 94 consecutive pairs have median 15.488 us, p95
16.352 us and maximum 16.480 us between rounds. Adaptive execution covers
542 rounds at widths 1/2/4/6 (507/33/1/1); its 498 consecutive width-one pairs
have median 15.296 us, p95 16.128 us and maximum 16.320 us. Both traces contain
no execution-time allocation, recapture, module loading, synchronization API
or out-of-graph compute. The graph interval remains close to the previous
package, but the 12.248 ms kernel span is higher than its 11.597 ms trace;
this qualification makes no operator-speed improvement claim. Full-model
throughput, literal zero idle and physical PPU execution remain unqualified.

A further SDK 2.1.1 host audit finds hg-prefixed declarations for all 66 entry
names resolved unconditionally by the native package driver. Compiled header
probes compare 70 structure sizes, alignments and field offsets with CUDA,
including graph kernel/copy parameters, VMM properties and profiling callbacks;
all match on x86-64. Thirty-five driver and callback constants also agree with
the native runtime's values. These checks do not establish device behavior.

The SDK's `CUDA_SDK/targets/x86_64-linux/lib` does contain `libcuda.so.1` and
`libcupti.so` facades. They forward to `libhggc.so` and `libhgpti.so`; the former
is absent from this local SDK installation. The facade exports all 66 package
driver entry names and all 49 names used by the asynchronous runtime. This
resolves the library-name question but does not qualify the underlying driver.
The checked HGPTI library requires GLIBC 2.38; a private compatible loader was
used for its host API checks without replacing system libraries.

Actual `cuptiGetCallbackName` calls through that facade return 45 expected
`hg*` names unchanged. The recorder now normalizes those names before its
existing module, nested-callback and allocation-lifetime handling. Fourteen
host recorder tests pass, including HG-named nested materialization, unload,
failed allocations and mapped-host lifetimes. After rebuilding the extension,
four RTX tests pass for real driver allocations/context destruction and native
graph export. This is not PPU graph replay qualification.

Callback 742 (`cuFuncLoad`) returns `INVALID_PARAMETER` through the actual
SDK2.1.1 HGPTI facade, consistent with its absence from the header, even though
`hgFuncLoad` is declared. Unsupported callback registration remains fatal;
its error now identifies the callback domain and ID. Resolve this requirement
with the deployment's profiling library before target export. The later local
SDK2.2.1 facade resolves `cuptiGetCallbackName(1, 742)` to `hgFuncLoad`; recorder
startup on the real device remains a separate check. Export naming now preserves
GB10/GB202 aliases and gives other valid NVML PCI identities a canonical
`pci-<vendor>-<device>` label. The box's eight `1ded:6001` devices use
`pci-1ded-6001` independent of their bus locations. New device IDs need no
naming-table update; conversion, acquisition and cache naming retain the signed
identity. This is not an executable compatibility check: operator targets,
runtime architecture checks and driver module loading still apply.

Native PPU ELF global discovery is now prepared for the observed machine 270
format. The old CUDA section-name filter returned no globals for a real PPU
module containing persistent counters, constants and a device pointer. The
exporter now enumerates allocated `.data`, `.bss` and `.rodata` objects for that
machine, retaining local objects as well as public ones. Existing native module
lookup, download and relocation validation still apply and must pass on PPU.
Eleven host tests pass, including the actual SDK-compiled fixture and preservation
of code, `.hg_info`, symbol tables and `.rela.data` through debug stripping.
No PPU kernel was launched; target module lookup and export/replay remain unrun.

The independent `qualify.ppu_export` gate compiles this fixture with native HGCC
on the deployment platform and prepares a four-case module-state sequence for
standalone Rust replay. It uses the production module recorder, VMM allocator
and package exporter, with no model or sampling dependency. Prepared graph
admission receives the explicit validated PPU operator target; context, handles
and buffer bounds still use the normal checks. The Rust allocation observer
recognizes both CUDA and HGPTI callback names and requires observed startup
allocations. See [the box resume commands](ppu-box-resume.txt) for the offline
module stage, native extension rebuild and remaining device export qualification.

Reproduce the offline fixture check with the selected SDK on PATH:

```sh
ppu_export_check=/workspace/ppu-export-check
mkdir -p "$ppu_export_check"
hgcc -arch=ppu_10 -x hg -std=c++17 -O2 -c \
  tests/preparation/fixtures/ppu_module_globals.cu -o "$ppu_export_check/globals.o"
(cd "$ppu_export_check" && hgobjdump --extract-elf=0 globals.o)
FLARE_TEST_PPU_ELF="$ppu_export_check/globals.o_ELF_File_1_PPU10" \
  .venv/bin/python -m pytest -q tests/preparation/test_cubin.py
```

### Native Qwen serving

Enable hybrid history drafting at serving startup with
`flare run PACKAGE --history-drafts`; disable it with `--no-history-drafts`.
These options use the package's existing MTP graphs and are not export settings.

Export Qwen with up to five MTP draft tokens per round:

```sh
.venv/bin/ignition export qwen3.8-flash --gpus 2
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/qwen38-tp1-mtp5.safetensors

.venv/bin/ignition export qwen3.8-flash --gpus 0,1
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/qwen38-tp2-mtp5.safetensors
```

Qwen defaults to 32 concurrent requests, five MTP draft tokens, and 128 target-token
rows, with explicit flags and configuration taking precedence. Verification is
rectangular: the default catalog covers five drafts per request through batch 16
and three drafts at batch 32. Already exported packages retain their captured
capacities.

`ignition export qwen3.8-flash --gpus 0,1 --shard-hyperconnections` partitions
the 320-wide hyperconnection bottleneck across ranks, including target and MTP
final mixers. Each rank loads its down-projection rows and up-projection columns;
normalization and residual injection stay replicated. Partial gate logits round
to BF16 before transport, and the reduced BF16 logits feed sigmoid and the stream
mean. This changes reduction rounding relative to the replicated projection.
The option requires TP2 or TP4 and is recorded in the package's rebuild metadata.
Replicated execution is the default: the compute savings must outweigh the extra
10,240-element reduction per mixer and token. Changing the option requires a
fresh export; `--reuse-weights` requires matching prepared weight layouts.

Vision recipes accept `--image-concurrency N`, independently of
`--max-concurrency`. It defaults to `min(4, max_concurrency)` and can also be set
as `image_concurrency` in `[defaults]` or a model's configuration section.
Explicit flags override configuration. Text-only recipes ignore the configured
image default and reject an explicit image-concurrency flag.
Helios reserves a feature slot when a request needs vision computation. Requests
whose image spans are already covered by a reused prefix checkpoint need no
feature slot. Requests needing a slot queue when those slots are occupied;
eligible requests continue to admit. A reserved slot remains owned until
request retirement, including completion of outstanding GPU work after cancellation.
The limit does not change images per request or the prepared patch capacities.
Changing it requires exporting the package again.
Packages with vision require the prepared feature-slot capacity and binding.
Images replay one at a time into bounded windows in the request's feature buffer.
Chunked prefill consumes each window before the buffer is reused. Combined image
tokens may exceed the buffer's capacity; each image must fit one window and a
captured patch capacity. HTTP body and total context limits also apply.
PNG, JPEG, and static WebP data URLs are accepted.

For Qwen, each feature slot holds 16,384 BF16 vectors of width 2,560: 80 MiB per
rank. Four image slots therefore reserve 320 MiB even with 32 active request slots.

N-gram tables use indirect delivery exclusively: every graph consumes compact
GPU row buffers filled by Helios. Qwen PLE tables are stored once in the package.
Helios reads the 26.8 GiB table from disk by default, sharing the reader and
row cache across local GPUs;
`flare run --embedding-table-storage ram` keeps it in resident host memory.
The graph publishes GPU-produced lookup IDs before layer
0 and waits for row delivery at the layer-1 consumer. Disk and RAM delivery
use the same prepared model graphs; the flag selects table storage only.

## GLM 5.3 Flash

The `glm-5.3-flash` recipe uses `nvidia/GLM-5.3-Flash-FP4-Local` and supports
one, two, or four GPUs, including TP1 on DGX Station. Its logical context is
1,048,576 tokens; the default 4096-token prefill chunk controls graph execution
size independently of that context.

Projection formats follow the checkpoint's ModelOpt declarations. The target
decoder uses MXFP8 for KDA Q/K/V, dense MLPs, and the vocabulary head;
block-FP8 with FP32 scales for MLA projections and shared experts; and BF16 for
recurrent gates, indexer projections, and MLA key/value absorption. Compatible
projections are fused within each precision group. Normalization retains BF16
or FP32 checkpoint weights. Learned drafts use the target MXFP8 vocabulary head.

Image preparation includes the checkpoint's replicated vision tower and native
image processor. The tower merges four patches into each image feature, with
captured capacities from 64 through 32,768 patches. Each feature slot holds an
8192-feature window; the shared scheduler reuses that window as image prompts
advance. `--image-concurrency` defaults to the smaller of four and text
concurrency. `--language-model-only` omits image weights and graphs.

## DeepSeek V4.1 Flash

The `deepseek-v4.1-flash` alias selects `deepseek-ai/DeepSeek-V4.1-Flash-1017`
and supports TP3 and TP4 text generation, reasoning, DSML tool calls, and images
when the checkpoint contains its vision encoder.
TP3 partitions the 2304-wide routed and shared experts into 768-wide shards
without padding. Attention partitions whole output groups: padding eight groups
to nine gives each rank three groups and 24 heads. The last rank's extra group
has zero query and output weights; all checkpoint weights and scales remain
unchanged. Vocabulary shards contain 43136 rows each; the 128 extra gathered
logits are excluded before sampling. Engram's projection input columns divide its 24 heads evenly across three
ranks. Every host stores full Engram tables in checkpoint row order; local ranks
share table reads and bounded host caching. Graphs request canonical table rows,
and each rank selects its head columns for the partial projection before reduction.
Exports use the native DeepSeek V4.1 prompt encoder and include only the tokenizer
asset. Vision exports bind the atomic `<｜deepseek_image｜>` token as ID 129264,
which must agree with checkpoint `image_token_id`. Packages without explicit
prompt-encoder metadata continue to use their tokenizer and Jinja template.
Sampling defaults are temperature 1.0 and top-p 1.0, following the
[model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017#minimal-inference).
Target vocabulary projections keep the checkpoint's BF16 weights and produce
FP32 logits for generation, prompt scoring, and cached-prefix resume. Proposal
head precision is configured separately with `--draft-head-quantization`.
Preparation losslessly compresses backbone and DSpark MXFP4 expert scales with
CSF by default. Compressed scales stay resident, with shared
expansion buffers reused by serial expert layers. Planning conservatively estimates
3% less resident weight storage; loading measures the actual allocation.
`--expert-scale-compression native` preserves native scales. MLA arithmetic defaults
to BF16; `--mla-compute-mode auto` lets b12x select arithmetic while preserving
packed KV storage.
DSpark is its learned draft method. Preparation defaults to seven draft tokens,
32 concurrent requests, a 4096-token prefill chunk, a 128-row physical target
capacity, and a budget of 64 packed round graphs. Positive-depth DSpark uses
packed GPU prefix selection automatically.
Each captured round contains proposal generation, GPU prefix allocation, target
verification, and accepted-state commit. Width-one graphs retain draft context
without generating proposals. Ordinary graphs remain available for native
`--mtp-draft-tokens 0`; `--no-mtp` omits draft weights during preparation.
Standalone proposal, verification, and joined wider speculative graphs are not
registered in the packed catalog. Image preprocessing preserves the checkpoint's
row-major patch layout and 1024-token image limit.

`--joint-tuning` enables joint selection of QKV projection, query normalization,
and query expansion plans. It covers every prepared text execution shape,
including prefill, ordinary decode, width-one continuations, packed target
verification, and DSpark proposals. Matching operator declarations share a
selection scored across their layer contexts; distinct declarations are tuned
independently. Batch capacity, physical token capacity, attention mode, and mHC
seams determine the measured contexts. The option defaults to off and requires
MXFP8 query normalization on the selected GPUs and `--autotune-effort default`
or `audit`. It adds preparation time without changing the serving graph catalog.

`--speculative-token-capacity` bounds physical target rows;
`--speculative-graph-budget` independently controls the number of packed rounds.
For each power-of-two request bucket `B`, the catalog retains capacities
`B + 1` and `min(token_capacity, B * (draft_tokens + 1))` when that interval is
nonempty. These endpoints preserve the smallest positive aggregate draft budget
and full-width verification wherever it fits. Width-one continuations cover
every request bucket independently and do not consume the packed graph budget.

Additional graphs refine the largest ratio between adjacent capacities across
all buckets. Each refinement inserts the integer nearest the geometric midpoint
that minimizes the larger resulting ratio. Equal ratios prefer smaller batches,
then smaller capacities. Comparing ratios gives each batch the same relative
resolution rather than assigning all graphs to the largest absolute row range.
At a fixed row cap, increasing the graph budget adds one distinct graph at a
time and preserves every graph already selected. Integer row capacities limit
the available resolution: larger budgets saturate at every valid capacity.
Budgets below the endpoint count fail before preparation; the default request
and token capacities require at least 12 packed graphs.

With seven drafts, 128 target rows, and a 64-graph budget, the catalog contains:

| Captured request capacity | Packed graphs | Minimum / maximum target rows |
|---|---:|---:|
| 1 | 7 | 2 / 8 |
| 2 | 10 | 3 / 16 |
| 4 | 12 | 5 / 32 |
| 8 | 12 | 9 / 64 |
| 16 | 14 | 17 / 128 |
| 32 | 9 | 33 / 128 |

These 64 combined packed rounds and six width-one continuations are separate
from ordinary decode, prefill, maintenance, and vision graphs. The persistent
preparation catalog prints every selected capacity. At batch 32, the 128-row
cap permits four target rows per request on average, while proposal generation
still produces seven drafts per request. Raising the row cap permits longer
aggregate verification prefixes; raising the graph budget improves resolution
within that cap and increases capture and retained graph resources.

The host owns admission, cohort membership, and cache reservations. The captured
round assigns prefixes from native proposal confidence and reports one terminal
completion. Native scheduling compares complete measured round cost and expected
emitted-token value across the packed and width-one candidates. Prefix limits
include output and context budgets, grammar constraints, and host-stop barriers.
Physical target padding executes at the captured row capacity.
Its 188.8 GiB of Engram tables are stored once in the package in checkpoint
row order. Every host needs the complete tables; local ranks share their
reader and host cache. Projection input columns are assigned round-robin by
hash head, so each TP4 rank computes two bigram, two trigram, and two four-gram
head contributions and reduces its partial projection before residual mixing.
Table rows and projection scales retain checkpoint precision.
Disk storage reads rows on demand through bounded direct-I/O staging;
`flare run --embedding-table-storage ram` loads the complete tables into
ordinary resident memory. Both modes copy compact rows into each rank's prepared
buffers and signal captured consumer events. Preparation appends the immutable
checkpoint spans without materializing the tables.
For disk storage, `--embedding-table-resident-scales` retains the original scale
bytes in RAM and removes scale reads on row-cache misses. This is enabled by
default on discrete GPUs and disabled when any selected GPU shares host memory.
Use `--no-embedding-table-resident-scales` to disable it explicitly. The scale
planes are shared across local ranks and add to the host's row-cache budget.
The loader reserves this memory before sizing KV storage.
Export with the recipe defaults on four available GPUs, then optionally create
a standalone packed file and serve it:

```sh
.venv/bin/ignition export deepseek-v4.1-flash --gpus 0,1,2,3
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/deepseek41-tp4.safetensors

./flare/target/release/flare run \
  /data/flare-runtime/deepseek41-tp4.safetensors \
  --gpus 0,1,2,3
```

For four DGX Sparks with one GPU per host, replace `--gpus 0,1,2,3` during
export with `--gpus 0 --ranks 4 --rendezvous 10.200.0.12:29630` and start an
`ignition join` process on each peer. Rank zero retains the checkpoint; peers
request their weight ranges during preparation. From a paired controller, use
`flare orbit run maxwell:/data/flare-runtime/deepseek41-tp4.safetensors --cluster sparks --gpus 0`
with a four-node `sparks` cluster. The prepared model uses RoCEnante collectives
across hosts. See [multi-host preparation and serving](multi-host.md) for
preparation authentication, orbit setup, and rank placement.

For three Sparks, use `--ranks 3` and join peer ranks 1 and 2. Launch the TP3
package with `flare orbit run NODE:PACKAGE --nodes NODE,PEER1,PEER2 --gpus 0`,
listing hosts in preparation rank order.

On GB10, a TP3 package with 32 concurrent requests, four image slots, seven
DSpark drafts, BF16 MLA, and CSF expert scales occupies about 104.7 GiB per
rank before physical cache backing. Use `--memory-limit=-1G` when launching
this configuration to retain a 1 GiB reserve. The integrated-memory planner
also preserves its separate 5% system allowance and 2 GiB host allowance.
A 4 GiB reserve leaves insufficient memory for continuation rows. Disk-backed
Engram tables retain the default 1 GiB row cache per host.

`--image-concurrency` defaults to the smaller of four and `--max-concurrency`.
For example, add `--max-concurrency 64` during export to
prepare 64 text request slots while retaining four independent image feature slots.
The native server can lower text concurrency at load time. Engram histories
exclude image positions, so n-grams cannot cross an image boundary.

DeepSeek V4.1's logical cache pages cover 128 tokens across compressed latent and
index regions with different byte sizes. Continuation rows retain sliding
windows, pending compression state, partial pages, DSpark context windows, and the
last hidden state. Both use runtime-sized virtual ranges; exported packages
contain their layouts and capacity bounds without cache payloads.
Engram indexing runs in the graph over committed history and candidate tokens.
The three preceding tokens live in continuation state, including reused prefixes.
DSpark proposals, target verification, and accepted-state commit use the shared
native speculative contract; Helios contains no DSpark model implementation.

The optional preparation flags `--draft-head-quantization nvfp4` and
`--markov-head-quantization nvfp4` select independent NVFP4 copies of the draft
vocabulary head and Markov output projection. Both use BF16 activations; the
Markov kernel's BF16 output is widened to FP32 before bias addition. Target
verification retains the checkpoint head. Omitting these flags preserves the
checkpoint precision. Each selection requires tuning, capture, and package export;
it cannot change an existing package at serving time. The same options apply to
the DeepSeek V4 Flash recipe. Quantization changes proposal probabilities and can
change acceptance and throughput; measure both separately and together.

DeepSeek V4.1 also accepts `--draft-head-quantization mxfp8` for an E4M3
proposal head with one power-of-two scale per 32 weights. The recommended
default is checkpoint precision: omit `--draft-head-quantization` to select it.
Use `--draft-head-quantization mxfp8` when reducing draft projection bandwidth
is useful while retaining BF16 activations, target checkpoint weight storage,
and the checkpoint Markov projection. It retains a separate quantized copy in
addition to the target weights, using more GPU memory. The initial weight
estimate excludes the extra copy;
preparation reports the actual allocation and sizes cache backing after loading.
It changes the proposal distribution and seeded text;
verification must use the actual proposal probabilities. Rebuild the package
when changing this preparation option.

```sh
.venv/bin/ignition export deepseek-v4.1-flash --gpus 0,1,2,3 \
  --draft-head-quantization mxfp8
```

DeepSeek V4.1 also accepts `--markov-head-quantization fp8`. It stores the draft
Markov output weights as E4M3 values with one FP32 reconstruction scale per
vocabulary row. The flag is independent of `--draft-head-quantization`; omit it
to retain checkpoint precision. Both the fused batch-one and wider prepared
Markov paths form FP32 transition scores, and the proposal draw and retained
rejection distribution use those same scores. Target vocabulary weights remain
BF16 and target logits remain FP32. Row-scaled FP8 is lossy draft arithmetic,
so seeded text and acceptance may change. To prepare NVFP4 draft vocabulary
weights together with FP8 Markov weights while retaining the recipe's other
defaults, run from the repository root:

```sh
.venv/bin/ignition export deepseek-v4.1-flash --gpus 0,1,2,3 \
  --draft-head-quantization nvfp4 --markov-head-quantization fp8
```

Use an ordinary export when first changing either head's weight format.
`--reuse-weights` trusts the previous package's weight bytes, while
`--no-weights` requires the exact changed payloads to be cached already.

`--vocabulary-parallel-sampling` keeps DeepSeek V4.1 proposal and verification
probabilities on their vocabulary-owning GPUs. Only normalization statistics,
selected-token probabilities, and residual winners cross ranks. The default is
disabled. DeepSeek V4.1 requires TP4 for this option; its TP3 gather is unavailable.
The option requires positive learned draft depth and supports BF16 or
row-scaled FP8 Markov weights; NVFP4 Markov weights are unsupported. Target
vocabulary weights and logits retain BF16 and FP32 precision. Full softmax and
greedy targets are supported; top-k, top-p, and min-p filtering must remain
disabled, as in the recipe's default policy. Raw generation logprobs and grammar
constraints remain available. Block rejection changes seeded output relative to
the full-vocabulary serial sampler.

For a TP4, single-request configuration with seven drafts, run:

```sh
.venv/bin/python -m ignition export deepseek-v4.1-flash --gpus 0,1,2,3 \
  --max-concurrency 1 --mtp-draft-tokens 7 --speculative-token-capacity 8 \
  --prefill-chunk-size 4096 --expert-scale-compression native --mla-compute-mode bf16 \
  --draft-head-quantization nvfp4 --markov-head-quantization fp8 \
  --vocabulary-parallel-sampling --joint-tuning --collective-tuning runtime \
  --collective-ladder ignition/preparation/ladders/gb202_pcie_tp4.json
```

Use a fresh weight export when first enabling this option: the Markov head's
stored vocabulary partition changes. Complete-model serving has been exercised
at TP4 with one concurrent request; wider serving configurations still need
model validation.

## DeepSeek V4 Flash

DeepSeek V4 Flash uses `local-inference-lab/DeepSeek-V4-Flash-Vision-Exp-lossless-CSF`, including its
vision encoder and checkpoint-integrated DSpark model. The `deepseek-v4-flash`
alias prepares five draft tokens by default, with adaptive verification. It
accepts images within its image-token budget and has no offloaded embedding tables.
Exports use the native DeepSeek V4 prompt encoder, carry only the tokenizer asset,
and bind the atomic `<｜deepseek_image｜>` placeholder as token 129264.
Both native DeepSeek encoders default to thinking enabled and high effort, accept
`low`, `high`, `xhigh`, and `max`, and reject `drop_thinking`, assistant
continuation, and `add_generation_prompt=false`. DeepSeek V4 rejects numeric
effort. DeepSeek V4.1 also accepts integer scores from 1 through 100 in
`reasoning_effort` or `chat_template_kwargs.reasoning_effort`; these are prompt
hints rather than token budgets. See [reasoning controls](api.md#reasoning-sampling-and-usage)
for request examples and named-level mappings. System-prefix caching
derives reusable token prefixes from the official encoder's rendered conversations.
Previously exported Jinja packages retain their
packaged rendering behavior and require both tokenizer and template assets.
Use two or four available GPUs for preparation and serving:

```sh
.venv/bin/ignition export deepseek-v4-flash --gpus 0,1,2,3
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/deepseek4-tp4.safetensors

./flare/target/release/flare run \
  /data/flare-runtime/deepseek4-tp4.safetensors \
  --gpus 0,1,2,3
```

V4 uses 256-token logical pages containing 64 C4 latent records with an FP8
retrieval index, or two C128 latent records with dense history attention. Each latent page
stores FP8/BF16 payloads followed by its own scale footer. Continuation rows
preserve the complete 128-token compression interval, C4 overlap, sliding
windows, partial pages, and DSpark context. Only accepted tokens update the
continuation state. Bidirectional image spans stay within one prefill chunk;
the scheduler shortens a preceding text chunk when necessary. Image preprocessing,
feature delivery, text execution, and output parsing all run natively in Helios.
Packages with 128-token V4 pages are incompatible and must be re-exported.

## MiMo V2.6 Pro MOPD

MiMo V2.6 Pro MOPD uses `XiaomiMiMo/MiMo-V2.6-Pro-MOPD` with checkpoint
MXFP4 experts, block-FP8 dense projections, and BF16 draft and vision weights.
The `mimo-v2.6-pro-mopd` alias prepares text, images,
and seven DFlash draft tokens with adaptive verification. Audio is not prepared.
Select eight available GPUs:

```sh
.venv/bin/ignition export mimo-v2.6-pro-mopd --gpus 0-7
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/mimo-v26-pro-mopd-tp8.safetensors

./flare/target/release/flare run \
  /data/flare-runtime/mimo-v26-pro-mopd-tp8.safetensors \
  --gpus 0,1,2,3,4,5,6,7
```

Global attention uses paged KV storage; continuation rows retain sliding-window
KV, partial pages, and accepted DFlash context. DFlash consumes selected target
layer features and predicts an eight-position block containing the pending
token and seven proposals. Helios uses the shared speculative execution contract;
draft modeling and state updates are captured by Ignition. The vision encoder
publishes compact image features for causal text prefill. Loading with
`--mtp-draft-tokens 0` selects ordinary execution without draft-state updates.

## Step-5 Preview

Step-5 Preview uses `stepfun-ai/Step-5-Preview-NVFP4`. The
`step5-preview-nvfp4` alias prepares text, images,
and three causal MTP draft tokens with adaptive verification. Checkpoint NVFP4
routed experts use W4A4. Routed experts declared BF16 in checkpoint metadata
are converted to NVFP4 during loading and use b12x W4A16 with BF16 activations;
the converted weights are retained in captured graphs and exported packages.
Shared, dense, and MTP MLP gate/up weights are packed and converted to MXFP8
during loading. Target-model projections select BF16 or quantized activations
through autotuning. MTP projections retain BF16 activations to bound rounding
error through the predictor feedback chain. Down and attention projections remain
BF16, with FP32 residual accumulation. Paged and sliding-window
K/V, including MTP windows, use FP8 E4M3 with unit descales;
sliding attention consumes live candidates in BF16 before committing them to FP8 history;
Q/K normalization and rotary computation retain their BF16 rounding before the
FP8 store. With CSF scales and shared scratch, the concurrency-16/MTP3 graph
catalog, including vision, fits on four RTX PRO 6000 Blackwell GPUs. This
memory-constrained example limits speculative target rows to 16 and selects
fixed-width serving, with width-one execution at batches that cannot fit wider
verification. The recipe defaults are concurrency 32, 128 target rows, and
adaptive verification:

```sh
.venv/bin/ignition export step5-preview-nvfp4 --gpus 0,1,2,3 \
  --max-concurrency 16 --no-mtp-adaptive-verification \
  --speculative-token-capacity 16 --expert-scale-compression csf
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/step5-tp4-mtp3-c16-fp8kv.safetensors

./flare/target/release/flare run \
  /data/flare-runtime/step5-tp4-mtp3-c16-fp8kv.safetensors \
  --gpus 0,1,2,3
```

`--expert-scale-compression csf` compresses all Step-5 routed-expert block
scales losslessly during loading. Both checkpoint W4A4 experts and converted
W4A16 experts use CSF, retaining their activation modes. Serialized layers share
one pair of expansion buffers, whose contents are zero-initialized on export.
One transient loading pool reuses conversion scratch across layers and retires
after weight loading, retaining live weights for graph export. The separate
graph-capture pool remains alive. Vision, prefill, decode, and draft activations
reuse typed graph buffers. MTP feedback, sampling, metadata, and resume scratch
reuse a separate buffer pool across graph variants, with disjoint buffers within
each graph. Sequential predictors share their norm-concatenation output. Kernel
workspaces use the shared operator arena, with separate slices for overlapping
expert branches. Compiler-created intermediates use one shared CUDA graph pool.
The memory plan estimates 6% lower resident weight storage; actual memory depends
on encoded exceptions and is reported after loading. Larger concurrency requires
separate preparation and memory validation.

Sliding attention keeps a 512-token continuation window. Sparse attention uses
128-token logical pages containing FP8 keys and values and an FP8 retrieval
index for eight-token regions. Continuation rows also preserve partial index
regions and accepted MTP state. Accepted target tokens, hidden states, and embeddings
remain in continuation storage until the following round consumes them in three
packed predictor passes. Candidate KV never enters committed predictor windows.
Region scoring, selection, and attention use
b12x kernels. Attention output gating is fused into the direct output or split
merge store, preserving BF16 rounding of attention and sigmoid before their
product. SwiGLU clamps the gate after SiLU. Helios manages these resources
through the shared cache and speculative execution contracts. The perception encoder
prepares 504-pixel local crops and a 728-pixel overview. Serving defaults follow
the model card: temperature 1, top-p 0.95, no top-k filter, and neutral penalties.

One-shot TP reductions within the channel's byte capacity fuse the residual
update and zero-centered RMSNorm into the collective. A bounded resident grid
retains row ownership across ranks while processing multiple rows per block.
The collective preserves FP32 residual storage and BF16 normalized outputs.
Attention and dense MLP projections enter the fused
collective directly as BF16; routed MLP updates enter as FP32. Attention and dense
MLP reductions round to BF16 before residual addition; routed MLP reductions retain
FP32. MLP reductions consume the following layer's input-normalization weight.
The final target and predictor MLPs
retain their separate final residual update. DMA prefill and hierarchical
collectives use separate reduction and normalization kernels.

## Kimi K3

Kimi K3 uses the local `/data/models/Kimi-K3-MXFP4-CSF` checkpoint. The
`kimi-k3-mxfp4-csf` alias prepares text generation, reasoning, XTML tool calls,
and images on sixteen GPUs. It preserves MXFP4 experts
with compressed scales, BF16 projections, and FP32 KDA recurrent state.
Its 93-layer decoder combines KDA, unrotated latent attention, attention
residuals, and latent SiTU experts. Red Hat DSpark uses
`RedHatAI/Kimi-K3-speculator.dspark` revision
`38a88101e0d46bb22134b9da340f381b954d40d4`, downloaded to the Hugging Face hub
cache (or `[paths].hub_cache`), and defaults to seven draft tokens per round.
A target checkpoint may instead carry that draft in its `dspark/` directory;
`kimi-k3-dspark` selects this layout at `/data/kimi-k3-dspark`.

For Kimi K3, `--online-mxfp8` converts KDA input, forget-gate, and output
projections; MLA latent/gate, query, and output projections; shared
expert gate/up and down projections; and both tensor-parallel layouts of the
MoE latent down projection and its latent up projection. Activations remain
BF16. MLA's batched key/value projections, routers, the first dense MLP,
embeddings, vocabulary head, and vision encoder retain checkpoint
precision. DSpark converts its context projection, attention outputs, and MLPs.
Shared and routed experts keep disjoint
kernel scratch when their execution overlaps.

This example limits concurrency to four and enables online MXFP8 conversion;
image concurrency follows the four-request limit:

```sh
.venv/bin/ignition export kimi-k3-mxfp4-csf --gpus 0-15 \
  --max-concurrency 4 --online-mxfp8
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/kimi-k3-mxfp4-csf-tp16-c4-images-redhat-dspark-7-mxfp8.safetensors

./flare/target/release/flare run \
  /data/flare-runtime/kimi-k3-mxfp4-csf-tp16-c4-images-redhat-dspark-7-mxfp8.safetensors \
  --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

The draft consumes AttnRes mixtures entering target layers 24, 48, 72, 88,
and 92, before input normalization. Its five causal Qwen3 layers use
2,048-token BF16 key/value windows. An anchor followed by mask tokens produces
seven proposals; each preceding proposal conditions the Markov bias.
Ignition samples proposals and uses adaptive target verification widths,
without the draft's confidence head. The
[Red Hat model card](https://huggingface.co/RedHatAI/Kimi-K3-speculator.dspark)
reports eight-token probabilistic proposals with block rejection sampling,
which differs from this execution configuration. Only accepted target features
update draft windows; KDA verification commits the accepted recurrent prefix.
Loading with `--mtp-draft-tokens 0` selects ordinary execution; exporting with
`--no-mtp` excludes draft weights and state.

Target FP8 MLA latent records occupy 128-token pages.
With context parallelism across sixteen ranks, concurrency four, and a
4096-token prefill chunk, preparation backs ten logical KV pages and nine
continuation rows: one null row, four live rows, and four snapshot destinations.
Retaining CUDA graph handles for export does not expand this working set.
Native package loading sizes serving caches from available device memory;
in-process diagnostic serving limits admission to physically backed storage.
Continuation rows preserve KDA
convolution and recurrent state, DSpark key/value windows, partial latent pages, and the normalized
last hidden state. The MoonViT encoder publishes hidden-column-sharded features
for causal text prefill, with storage for 65,536 patches per active feature
window. Conversations can consume multiple windows through chunked prefill.
XTML encoding distinguishes protocol tokens from literal marker text.
Serving defaults follow the model card's agentic sampling recommendation:
temperature 1, top-p 1, no top-k filter, and neutral penalties. Reasoning is
always enabled; requests to disable thinking are ignored without an error.
Effort defaults to `max`; requests can select `low`, `high`, or `max` and
override sampling defaults. Preserve assistant reasoning in conversation history.

## MiniMax M3.1 Flash

The `minimax-m3-1-flash` alias selects
[`MiniMaxAI/MiniMax-M3.1-Flash`](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)
at revision `9e7ebacac350497ad4a1272a2e817639bf11561c`. Preparation uses four
SM120/SM121 GPUs, MXFP8 projections, NVFP4 routed experts, and FP8 target,
indexer, and DSpark caches. Sampling defaults are temperature 1.0 and top-p
0.95, following the model card.

DSpark defaults to 32 concurrent requests, five draft tokens,
a 128-row physical target capacity, a 64-graph packed budget, and GPU causal
prefix allocation. Each packed graph generates proposals, allocates prefixes,
verifies the admitted tokens, and commits accepted state. Width-one graphs
preserve draft context. `--speculative-token-capacity` and
`--speculative-graph-budget` control the same shared catalog geometry described
under DeepSeek V4.1.

Ignition applies `ignition/recipes/minimax_m3/assets/chat_template.patch.json`
to the downloaded checkpoint template. The annotated result preserves the pinned upstream template byte for byte after removing message
marks. It retains the upstream adaptive-thinking behavior and XML tool syntax.
Vision preparation includes temporal patch duplication and spatial patch
merging; `--language-model-only` omits the vision encoder. Optional
`--expert-scale-compression csf` losslessly compresses expert scales and shares
their expansion scratch across routed layers. Planning retains the native
weight budget for this encoding; loading reports its actual allocation.

```sh
.venv/bin/ignition export minimax-m3-1-flash --gpus 0-3
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/minimax-m3.1-flash-tp4.safetensors
flare/target/release/flare run /data/flare-runtime/minimax-m3.1-flash-tp4.safetensors --gpus 0,1,2,3
```

## Muse Spark Open

The `muse-spark-open-nvfp4` alias selects
[`meta-models/Muse-Spark-Open-816B-A42B-NVFP4`](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4)
at revision `4239e4db19376904f2e82c167e78f70abb5455fe`. Text preparation uses
eight SM120 GPUs, checkpoint NVFP4 experts, FP8 E4M3 KV storage, a 262144-token
context, four concurrent requests, and 4096-token prefill chunks. Target and draft
queries and attention outputs use BF16. Image, audio, and video are not prepared.

DFlash2 drafting uses
[`nvidia/Muse-Spark-Open-816B-A42B-DFlash2`](https://huggingface.co/nvidia/Muse-Spark-Open-816B-A42B-DFlash2)
at revision `fbec887a79866e9ecff6c57adfd9fd02aa3cadea`. Preparation downloads
its configuration and weights into the configured Hugging Face hub cache. The
five-layer BF16 draft uses eight-token blocks with seven proposed tokens, target
features from layers 3, 19, 31, 47, and 59, and 2048-token sliding attention.
Target and draft histories use unit-scale FP8 E4M3 storage. The default adaptive
verification catalog uses width-one and width-eight graphs with per-request
prefixes of one through eight tokens and 32 packed target-token rows.
`--mtp-draft-tokens N` selects zero through seven proposals;
`--no-mtp-adaptive-verification` fixes the verification width. `--no-mtp` skips
the draft download, weights, state, and speculative graphs.

At tensor parallelism eight, draft weights occupy 1,596,065,024 bytes per rank,
including replicated convolution and candidate-selection tensors and fused context
projections. Each continuation row occupies 27,525,120 bytes with draft history
or 24,903,680 bytes with `--no-mtp`; each 128-token paged KV block occupies
524,288 bytes. These cache sizes exclude transient graph scratch.

Sampling follows the checkpoint model card: temperature 1.0, top-p 1.0, and
top-k disabled. The checkpoint generation configuration's top-k 64 is not the
recommended inference default. Reasoning effort accepts `low`, `medium`,
`high`, and `xhigh`, with `xhigh` as the template default. Set
`reasoning_effort` or `chat_template_kwargs.reasoning_effort` explicitly to
override it.

Muse Spark preparation requires FlashInfer/b12x operators for unweighted NVFP4
routed-expert outputs, independent FC1 half gains, and head-dimension-64 paged
decode with BF16 queries and FP8 E4M3 caches, including quantized sliding-cache
verification and rollback. The FlashInfer Git dependency pinned in
`pyproject.toml` lacks these operators. Complete the standard environment setup and install the local
checkout's build and runtime requirements into `.venv` first. Build requirements
include `setuptools>=77`, `packaging>=24`, and `apache-tvm-ffi>=0.1.11,<0.2`;
the CUDA 13 checkout's NCCL-EP dependency requires `nvidia-nccl-cu13>=2.30.7`.
Install the checkout containing the required operators with:

```sh
FLASHINFER_BUILD_NO_PIP=1 uv pip install --python .venv/bin/python \
  --no-deps --no-build-isolation -e /path/to/flashinfer
```

`FLASHINFER_BUILD_NO_PIP=1` disables nested dependency installation and NCCL
upgrades in the FlashInfer build hook. `--no-build-isolation` uses the build
requirements already installed in `.venv`.

Running `uv sync --locked --extra prepare --extra test` restores the pinned Git
dependency, so repeat the editable installation afterward when preparing Muse
Spark. The local checkout must provide both the `flashinfer` and `b12x` imports.

Ignition applies `ignition/recipes/muse_spark/assets/chat_template.patch.json`
to the downloaded checkpoint template. The annotated result preserves the canonical checkpoint bytes when message marks are removed.
Native recipient headers route `self` messages to reasoning, `commentary` and
`user` messages to visible content, and tool recipients through ATEM function
calls. `<|eom|>` separates messages within a generated turn; `<|eot|>` and
`<|end_of_text|>` end generation. Tool arguments in request history are normalized
to JSON objects before rendering. Caller content and header attributes retain
the canonical control-sequence escaping.

```sh
.venv/bin/ignition export muse-spark-open-nvfp4 --gpus 0-7 --prompt-logprobs
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/muse-spark-tp8.safetensors
flare/target/release/flare run /data/flare-runtime/muse-spark-tp8.safetensors --gpus 0,1,2,3,4,5,6,7
```

## Storage and graph capacities

Gemma 4 and Muse Spark default to four concurrent requests. Gemma 4's speculative
target-token capacity is the larger of padded concurrency and draft count plus
one, giving six rows at its default settings. Muse Spark declares 32 packed
target-token rows for seven DFlash2 drafts at concurrency four. Other recipes
default to 32 concurrent requests, with speculative recipes declaring 128 target-token rows independently
of concurrency.
Explicit flags and configuration override these defaults; existing packages retain their
captured capacities. Memory planning rejects capacities that do not fit the
selected GPUs.

| Recipe | Draft tokens | Verification layout | Speculative graphs at default concurrency | Maximum drafts at batch 32 |
|---|---:|---|---:|---:|
| Qwen3.8 Flash, Qwen3.8 27B, GLM 5.3 | 5 | Rectangular | 34 | 3 |
| Step-5 | 3 | Rectangular | 24 | 3 |
| DeepSeek V4, MiniMax M3.1 | 5 | Packed | 64 packed + 6 width-one | Up to 5, sharing 128 rows across requests |
| DeepSeek V4.1, MiMo V2.6 Pro, Kimi K3, Nemotron Bolt | 7 | Packed | 64 packed + 6 width-one | Up to 7, sharing 128 rows across requests |
| Gemma 4 | 5 | Rectangular | 10 at concurrency 4 | Not prepared at default concurrency |
| Muse Spark | 7 | Packed | 49 packed + 3 width-one at concurrency 4 | Not prepared above concurrency 4 |

The 128-row rectangular catalogs cover full draft depth through batch 16.
Gemma 4's six-row default covers five drafts at batch 1 and two at batch 2,
with width-one execution at batch 4. The counts include
width-one speculative graphs and exclude ordinary decode, prefill, maintenance,
and vision graphs. Capturing full rectangular depth at batch 32 requires 192 rows
for five drafts or 256 rows for seven drafts, set with
`--speculative-token-capacity`. Every packed recipe's round graph count is bounded
independently by `--speculative-graph-budget`.

Every model export prints a VRAM inventory for each rank after final publication,
including exports without `--verbose`. Heliopause measures device usage and package
allocator mappings; Ignition attributes retained ranges to checkpoint tensors,
scratch pools, graph temporaries, I/O and persistent state, communication buffers,
kernel globals, and allocator slack. Owner fields and defining source locations
identify the largest consumers in each category. Aliased ranges count once;
shared scratch displays representative tensor owners.

Preparation-only storage is separate from fixed package storage. Cache figures
count physically backed pages, not their virtual serving capacity. Mapped host
storage and imported peer mappings are listed outside local VRAM totals. PyTorch
allocated and reserved bytes overlap the attributed storage. Cubin data sizes are
not resident code sizes: graph executable overhead, kernel code, other driver
allocations, and other processes remain in the unattributed device-usage remainder.
The returned `RuntimePackage.memory_inventory` contains every attributed source
group; console output shows the largest five per category. Source locations name
owning classes or closures, not allocation stack traces.

Ignition commands accept `--config PATH`; the default configuration is
`~/.config/flare/config.toml`. Storage locations are configured with
`[paths] hub_cache`, `kernel_cache`, and `package_cache`; `[memory] reserve`
leaves GPU memory available for other processes during preparation and serving.
Ignition additionally budgets 1 GiB for warm-up and capture, plus 128 KiB per
effective prefill token above 4096, including mixed prompt/decode rows.
Preparation KV backing covers a prefill chunk plus boundary pages per captured request row, unless an
explicit preparation page count is supplied programmatically. Continuation
backing covers the null row, live request and verification rows, and snapshot
destinations used during preparation. Planned prefix-cache slots remain virtual;
they do not require physical backing during export. The memory report separates
the declared continuation extent from `preparation_state_rows` and
`preparation_state_bytes` where those extents differ. Both caches retain their
full virtual extents in the package. Helios
sizes its caches after uploading the selected graphs, keeping a separate 64 MiB
driver margin.
At package load time, `flare run --kv-cache-limit context` bounds the shared
KV pool by one prepared context; `concurrency` uses one context per selected
active request, and the default `memory` fills the available budget. Runtime
`--memory-limit` accepts an occupied size (`30G`), free reserve (`-30G`), or
fraction of total device memory (`0.8`). It replaces the exported serving
reserve and includes model resources as well as caches. Both flags apply at
package load time without another export. See
[runtime memory limits](cli-design.md#runtime-memory-limits) for rounding,
minimum capacities, and integrated-memory accounting. Preparation retains its
own existing memory policy.

Before compilation and capture, preparation prints a persistent graph catalog
with counts, batch and prefill capacities, logical widths, physical target rows,
and draft depths, including when `--verbose` is omitted.
Ignition captures power-of-two
batch capacities through the concurrency limit
and width one at every bucket. Rectangular speculative and verification shapes
require `batch * width <= speculative_token_capacity`; the preparation flag
`--speculative-token-capacity` defaults to 128 physical target rows except for
Gemma 4's derived capacity. Packed recipes use the shared capacity selection
described under DeepSeek V4.1 rather than rectangular `batch * width` shapes.
Previously exported separate-proposal
catalogs cover every bucket and load-time draft depth independently of that limit.
Flare can reduce concurrency at load time. Rectangular catalogs permit smaller
positive draft limits; packed catalogs require their captured positive depth or
zero. Prefill capacity stays prepared.
With `--mixed-prefill`, Qwen also captures mixed prompt/decode graphs: every
prompt bucket at width one and one balanced prompt bucket at each wider
verification depth, for each batch capacity of at least two. The ordinary and
draft-state-maintaining catalogs remain distinct. Mixed capture is disabled by
default and independent of the MTP flags.
See [mixed graph coverage](scheduling-runtime-design.md).

## Serving runtime packages

Load an exported `.safetensors` package directly with the native CLI:

```sh
./flare/target/release/flare run /data/flare-runtime/qwen38-tp2-mtp5.safetensors \
  --gpus 0,1
curl --fail http://127.0.0.1:8000/health
```

Select a subset of a package's captured execution capacities:

```sh
./flare/target/release/flare run /data/flare-runtime/qwen38-tp1-mtp5.safetensors \
  --gpus 2 --max-concurrency 2 --mtp-draft-tokens 3 \
  --no-mtp-adaptive-verification
```

`--mtp-draft-tokens N` limits each round to N drafts plus its pending input.
`--mtp-draft-tokens 0` selects ordinary prefill, decode, and prefix-cache graphs
without draft execution or draft-state updates. Every speculative package also
contains these ordinary graphs. Adaptive width-one rounds retain draft state so
later rounds can resume speculation. Ignition's `--no-mtp` excludes the draft model
during preparation. `--mtp-adaptive-verification` selects among the loaded
widths; `--no-mtp-adaptive-verification` selects the fixed width where captured,
with width-one execution available at larger batch capacities. Omitted flags
use the package's serving defaults. Limits cannot exceed captured capacity;
missing required batch/width combinations fail before GPU initialization.
Concurrency three requires the batch-four bucket. Only selected graphs and
required kernel functions are loaded. Weights, MTP weights, and transport buffers
retain their prepared layouts. Continuation rows and KV pages occupy separate
fixed virtual ranges with declared maximum extents. Helios backs those ranges
according to selected concurrency, verification width, and available GPU memory;
reducing execution state leaves more physical memory for KV pages.

The selected GPU count must match the package, and each GPU must match its
rank's architecture and memory requirements. Flare serves all ranks in one
process with one execution thread per rank. Its dashboard runs on an
interactive terminal. Use `--api-key-file PATH` to supply a serving API key;
exported packages do not contain the key. Stop Flare with Ctrl-C or SIGTERM
so native replay drains before CUDA resources are released.

## DGX Spark

DGX Spark's GB10 is SM121 and shares physical DRAM with the CPU. Prepare an
SM121 package on Spark; an SM120 package does not satisfy the architecture
check. GPU weights, KV, registered host buffers, resident embedding tables,
compiler workers, and the OS compete for that memory. Set `[memory] reserve`
for host activity as well as driver headroom; `reserve = "4G"` leaves four GiB
outside the model's cache budget. RAM embedding tables load before cache
sizing. Disk embedding tables use bounded staging and can retain their scale
planes in RAM with `--embedding-table-resident-scales`. The loader accounts for
resident scales and the optional row cache before sizing KV, for both PLE and
Engram packages. Disk tables default to a 1 GiB host-RAM row cache per Spark;
`--embedding-table-cache-size` overrides it, and `0` disables it.

Spark's NVML interface does not report VRAM capacity. Ignition identifies an
integrated device through CUDA and uses Linux `MemAvailable`, capped by CUDA's
device total, for planning. Qwen preparation and Helios use the same available
RAM budget when allocating weights and caches; swap is not GPU capacity.
CUDA free-memory queries exclude reclaimable page cache on GB10. Compiler worker
budgets account for both CPU and GPU allocations in shared RAM. Helios sizes
its caches independently after loading.
VMM allocations and registered host mappings remain distinct CUDA
memory types despite sharing DRAM; do not dereference a device allocation on
the CPU or replace system-scope synchronization with volatile accesses. See
[NVIDIA's Spark CUDA guidance](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html).

A model too large for one Spark runs tensor-parallel across two or four Sparks,
one rank per host, reducing through b12x RoCEnante over their ConnectX-7 ports:
`ignition export --ranks N --rendezvous ADDR:PORT` on rank zero with
`ignition join` on each peer writes one package on rank zero, and
`flare orbit run NODE:PACKAGE --cluster NAME` serves it from any machine,
starting every rank through its host's `flare agent`. See
[serving across DGX Sparks](multi-host.md) for agent installation, pairing,
and failures.

## Package and development serving

Export a package, then launch the native server:

```sh
.venv/bin/ignition export qwen3.8-flash --gpus 0,1
.venv/bin/ignition convert CACHE_ENTRY --form packed --output qwen.safetensors
flare/target/release/flare run qwen.safetensors --gpus 0,1
```

To test a serving-only Rust change, rebuild Flare and load an existing compatible
package. Re-export when changing prepared kernels, capacities, or the package
format. Export destinations must not already exist.

Use `.venv/bin/ignition export --help` for preparation flags and
`flare/target/release/flare run --help` for serving flags. See [the runtime package design](runtime-package-design.md)
for resource ownership and [the CLI design](cli-design.md) for configuration.

For fault diagnosis, `ignition debug-serve` hands prepared graphs directly to
Helios without exporting a package. Its native path requires Heliopause's
`qualification` Cargo feature. All debug modes, including `--direct` and
`--eager`, are unsuitable for actual serving or apples-to-apples performance
comparisons against other runtimes; use an exported package with `flare run`.
Select idle devices explicitly; see [the debug launch example](cli-design.md#development-without-package-export)
for the extension build and diagnostic commands.
