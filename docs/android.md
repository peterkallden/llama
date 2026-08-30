
# Android example client

The Android application under `examples/llama.android` is a development and
integration example, not a production-ready mobile client. It exists to test
the Android host boundary, SQLite-backed agent runtime, local model execution,
MCP transport, structured events and resource imports. Its UI and lifecycle
policies are intentionally small and may change as the host contracts mature.

## Build GUI binding using Android Studio

Import the `examples/llama.android` directory into Android Studio, then perform a Gradle sync and build the project.
![Project imported into Android Studio](./android/imported-into-android-studio.jpg)

This Android binding supports hardware acceleration up to `SME2` for **Arm** and `AMX` for **x86-64** CPUs on Android and ChromeOS devices.
It automatically detects the host's hardware to load compatible kernels. As a result, it runs seamlessly on both the latest premium devices and older devices that may lack modern CPU features or have limited RAM, without requiring any manual configuration.

The llama-agent Android development build packages two complementary backends
for `arm64-v8a`:

* the portable CPU backend, including runtime-selected ARM/NEON kernels; and
* the optional Vulkan backend, loaded dynamically when a compatible device and
  Vulkan driver are available.

The CPU backend remains the fallback. A physical device is needed to verify
Vulkan execution and performance; CI can still verify shader generation and
that both native backend libraries are produced and packaged. The host build
needs the Android SDK/NDK and Vulkan host headers (`libvulkan-dev` on Ubuntu).
The NDK supplies `glslc` and the SPIR-V headers used during shader generation.

The Android agent host currently selects the common CLI inference backend for
local GGUF execution. Android still provides the simpler daemon-like profile
through `AgentRuntimeService`: one configured GGUF generation model, one native
worker queue, structured events, cancellation and SQLite-backed state. The
desktop server-context backend and its server/mtmd dependencies are not part
of this Android target.

This is an embedded service profile, not the standalone Linux
`llama-agent-daemon` executable. Android does not currently include the Linux
daemon's inbound TCP, Unix-socket or HTTP listener transports, resident
multi-model loader, embedding-model path, training worker or multi-model
scheduler. It does include an outbound MCP HTTP client: once an endpoint is
configured, its discovered external tools are resolved into the normal Android
turn's tool catalog and can be selected by the model. Those boundaries must not
be silently implied by the common agent contracts.

The Android agent integration is a host around the common agent runtime. The
first functional path is a Service-owned local daemon profile with SQLite
state, structured events and cancellation; the Activity/UI is a client of
that Service. `AgentClientSession` is the example application's runtime adapter: it
binds the Service, configures an imported model, submits turns and polls
event/result messages. The JNI boundary stays small and does not expose
planner or tool implementation classes. GGUF models remain outside the APK in
app-private storage after import through `ContentResolver`.

Native worker threads should publish the existing agent events into a
host-owned queue. Kotlin can poll or collect that queue without receiving
callbacks from threads whose lifetime is controlled by the native runtime.

Model startup is asynchronous from the Activity's perspective. The Service
reports `loading`, invokes the common runtime's turn-free model preparation
seam, and reports `ready` only after the GGUF model and inference session have
been initialized. The example UI shows a circular progress indicator while
loading and keeps input disabled until that state is reached. Failed loading
is represented as `failed` rather than as a misleading ready state. The same
transition is also published through the common event queue as
`model_loading`, `runtime_ready` or `runtime_failed`.

`AndroidResourceStore` and `AndroidModelManager` reuse the existing path-based
agent seams. They import a selected `content://` URI through Android's
`ContentResolver` into app-private storage, then pass the resulting path to the
common runtime. GGUF files therefore remain outside the APK and Android-specific
URI handling does not leak into resource, planning or tool code.

The example client uses the same resource seam for user attachments. Selecting
one or more files with **Attach** imports them into the private resource store;
the next turn carries their imported paths as `resource_refs` in the shared
turn request. The native host converts those references to
`common_agent_input_resource` values, so normal resource discovery, MIME
processing and tool input binding remain common code. The original Android
`content://` URI is retained as import metadata, while native tools receive a
stable app-private path that they can actually read.

The **Attach** dialog lists imported files and supports selecting files with
checkboxes for removal. Model selection, attachment paths and the selected
session-side client state are restored across normal Activity recreation; the
native Service remains the owner of the runtime state.

The JNI facade also exposes the common MCP HTTP transport through
`configureMcp`, `mcpTools` and `mcpCall`. This reuses `agent_mcp_http_client`
and does not duplicate an MCP parser in Kotlin. Desktop uses the injected
`cpp-httplib` adapter; Android uses a small `HttpsURLConnection` adapter for
`https://` endpoints and therefore uses Android's normal certificate and
hostname validation. HTTP remains available through the native adapter for
local development. The Kotlin class only performs the platform request and
returns bounded status/body/header data; MCP protocol parsing, tool discovery
and tool calls remain in C++. These calls are a host transport seam, not a
second planner or tool registry.
When an endpoint is configured, the normal Android session `tooling_resolver`
now reuses the existing `mcp_agent_tool_provider` and exposes its resolved
tools to planning/execution. The capability snapshot reports both endpoint
configuration and HTTPS availability separately.

Credential handling follows the same host-boundary rule. MCP configuration may
keep an inline `bearer_token` for development and simple local setup, but
production Android configuration should use a `credential_ref`. The native
MCP client accepts an injected `common_agent_credential_provider`; Android can
resolve that reference through a Keystore-backed implementation, while desktop
hosts may use a keyring or another host-owned store. If both are configured,
the provider-resolved credential takes precedence. Secrets are not part of
model-facing tool contracts or runtime event output.

The example settings dialog can configure the MCP server name, HTTPS URL and
either a development bearer token or a host-owned `credential_ref`. The bearer
field is intentionally provided for local testing; production Android clients
should resolve credentials through the platform provider instead.

The library now includes a small `com.arm.aichat.agent.AgentRuntime` JNI
lifecycle facade. It owns a native handle, cancellation state and the common
event queue, and exposes `create(storageDirectory, modelPath?)`,
`submitTurn`, `cancel`, `resetCancellation`, `pollEvent`, `pollResult`,
`state` and `close`. `submitTurn` is non-blocking: a native worker invokes the
existing common session host, publishes the existing structured events, and
places the completed turn result in a separate result queue. Cancellation
belongs to one turn: after a stopped turn has fully returned, the Service may
explicitly reset it before starting the next turn. The model path is optional
so the Service can start before a model is selected; a turn is rejected until
one is configured. `state()` reports whether a model is configured, whether
the SQLite/session-host state opened successfully, and whether a turn is active.
Android owns lifecycle and transport; the common runtime remains the owner of
planning, tools, inference and event meaning.

`capabilities()` returns a host snapshot rather than a promise based on the
build alone. It reports the ABI, CPU/NEON support, whether the Vulkan backend
was packaged, whether Android can currently load `libvulkan.so`, whether
SQLite storage is open, and whether an MCP endpoint has been configured. A
configured MCP endpoint is not the same as reachable or authorized. A packaged Vulkan backend is not the same as a usable
Vulkan device; device enumeration and performance validation still belong to
the Android host/device smoke tests.

`com.arm.aichat.agent.AgentRuntimeService` is the Android lifecycle
owner for that facade. It is non-exported, returns `START_NOT_STICKY`, and
releases the native handle in `onDestroy`. An Activity or other UI component
may bind to it as a client. The Service does not become a second planner or
inference implementation; it owns process/lifecycle concerns and forwards
model configuration, submit, cancellation, capability, MCP transport, state,
event polling and result polling to the native runtime. Runtime replacement for
a newly selected model is also Service-owned. `close()` waits for an active
native turn to finish; callers should
request cancellation first when leaving the Service.

The example Activity pauses the native model runtime when it leaves the
foreground and resumes it when it returns. This releases model memory while
keeping the Service's storage directory, session identity and imported
resources intact. A running turn is cancelled before the pause; this is a
deliberate test-client policy and can later be replaced by a foreground-service
policy for long-running work.

The JNI result is deliberately a small transport contract rather than a
mirror of C++ classes. Its turn request, event envelope and terminal result
reuse the transport-neutral JSONL wire contracts under
`common/agent/protocol`; Android does not start the daemon or invent a second
turn protocol. The Android service may carry one JSON message per JNI call,
while daemon stdio/TCP adapters carry the same messages as newline-delimited
JSON. Events contain `type`, `detail` and only populated semantic identifiers.
Completed results contain `request_id`, `ok`, `cancelled`, `response`,
`plan_id`, `error`, `failure_class` and `event_count`.

The Android-specific verification is split at the same boundary. The native
`llama-agent-jsonl-wire-contract-smoke` checks the shared request/event/result
serialization and round-trip framing. Android instrumentation tests check
the Service manifest/lifecycle boundary and Android storage/resource imports;
they do not reimplement or snapshot the C++ wire codec in Kotlin. Emulator
tests that require a real model, CPU backend or Vulkan device remain optional
device tests and are not substitutes for the portable contract tests.

A minimal Android app frontend is included to showcase the binding’s core functionalities:
1.	**Parse GGUF metadata** via `GgufMetadataReader` from either a `ContentResolver` provided `Uri` from shared storage, or a local `File` from your app's private storage.
2.	**Obtain a `InferenceEngine`** instance through the `AiChat` facade and load your selected model via its app-private file path.
3.	**Send a raw user prompt** for automatic template formatting, prefill, and batch decoding. Then collect the generated tokens in a Kotlin `Flow`.

For a production-ready experience that leverages advanced features such as system prompts and benchmarks, plus friendly UI features such as model management and Arm feature visualizer, check out [Arm AI Chat](https://play.google.com/store/apps/details?id=com.arm.aichat) on Google Play.
This project is made possible through a collaborative effort by Arm's **CT-ML**, **CE-ML** and **STE** groups:

| ![Home screen](https://naco-siren.github.io/ai-chat/policy/index/1-llm-starter-pack.png)  | ![System prompt](https://naco-siren.github.io/ai-chat/policy/index/5-system-prompt.png)  | !["Haiku"](https://naco-siren.github.io/ai-chat/policy/index/4-metrics.png)  |
|:------------------------------------------------------:|:----------------------------------------------------:|:--------------------------------------------------------:|
|                      Home screen                       |                    System prompt                     |                         "Haiku"                          |

## Build CLI on Android using Termux

[Termux](https://termux.dev/en/) is an Android terminal emulator and Linux environment app (no root required). As of writing, Termux is available experimentally in the Google Play Store; otherwise, it may be obtained directly from the project repo or on F-Droid.

With Termux, you can install and run `llama.cpp` as if the environment were Linux. Once in the Termux shell:

```
$ apt update && apt upgrade -y
$ apt install git cmake libandroid-spawn
```

Then, follow the [build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md), specifically for CMake.

Once the binaries are built, download your model of choice (e.g., from Hugging Face). It's recommended to place it in the `~/` directory for best performance:

```
$ curl -L {model-url} -o ~/{model}.gguf
```

Then, if you are not already in the repo directory, `cd` into `llama.cpp` and:

```
$ ./build/bin/llama-cli -m ~/{model}.gguf -c {context-size} -p "{your-prompt}"
```

Here, we show `llama-cli`, but any of the executables under `examples` should work, in theory. Be sure to set `context-size` to a reasonable number (say, 4096) to start with; otherwise, memory could spike and kill your terminal.

To see what it might look like visually, here's an old demo of an interactive session running on a Pixel 5 phone:

https://user-images.githubusercontent.com/271616/225014776-1d567049-ad71-4ef2-b050-55b0b3b9274c.mp4

## Cross-compile CLI using Android NDK
It's possible to build `llama.cpp` for Android on your host system via CMake and the Android NDK. If you are interested in this path, ensure you already have an environment prepared to cross-compile programs for Android (i.e., install the Android SDK). Note that, unlike desktop environments, the Android environment ships with a limited set of native libraries, and so only those libraries are available to CMake when building with the Android NDK (see: https://developer.android.com/ndk/guides/stable_apis.)

Once you're ready and have cloned `llama.cpp`, invoke the following in the project directory:

```
$ cmake \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_C_FLAGS="-march=armv8.7a" \
  -DCMAKE_CXX_FLAGS="-march=armv8.7a" \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -B build-android
```

Notes:
  - While later versions of Android NDK ship with OpenMP, it must still be installed by CMake as a dependency, which is not supported at this time
  - `llamafile` does not appear to support Android devices (see: https://github.com/Mozilla-Ocho/llamafile/issues/325)

The above command should configure `llama.cpp` with the most performant options for modern devices. Even if your device is not running `armv8.7a`, `llama.cpp` includes runtime checks for available CPU features it can use.

Feel free to adjust the Android ABI for your target. Once the project is configured:

```
$ cmake --build build-android --config Release -j{n}
$ cmake --install build-android --prefix {install-dir} --config Release
```

After installing, go ahead and download the model of your choice to your host system. Then:

```
$ adb shell "mkdir /data/local/tmp/llama.cpp"
$ adb push {install-dir} /data/local/tmp/llama.cpp/
$ adb push {model}.gguf /data/local/tmp/llama.cpp/
$ adb shell
```

In the `adb shell`:

```
$ cd /data/local/tmp/llama.cpp
$ LD_LIBRARY_PATH=lib ./bin/llama-simple -m {model}.gguf -c {context-size} -p "{your-prompt}"
```

That's it!

Be aware that Android will not find the library path `lib` on its own, so we must specify `LD_LIBRARY_PATH` in order to run the installed executables. Android does support `RPATH` in later API levels, so this could change in the future. Refer to the previous section for information about `context-size` (very important!) and running other `examples`.
