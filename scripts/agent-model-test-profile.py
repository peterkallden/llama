#!/usr/bin/env python3
"""Run the existing model CTests serially with explicit setup and saved output.

Unlike ordinary CTest, this opt-in profile treats a skipped model test as an
incomplete validation. It never changes synthetic fixtures or runtime policy.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import threading


DEFAULT_TESTS = r"^llama-agent-(model-adaptation|flydelta-(concept-synthesis-model|model-ab|cvec-batch-model|runtime|repair-model|causal-patch-model|dataset-question(-repair)?-model))-ctest$"


def collect_device_samples(pid, stop, output):
    """Retain process-specific NVIDIA usage; lack of evidence is not CPU proof."""
    with output.open("w") as samples:
        while not stop.is_set():
            try:
                query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                if query.returncode == 0:
                    for line in query.stdout.splitlines():
                        if line.split(",", 1)[0].strip() == str(pid):
                            samples.write(line + "\n")
                            samples.flush()
            except (OSError, subprocess.TimeoutExpired):
                return
            stop.wait(2)


def trace_observed(path):
    with path.open(errors="replace") as output:
        return any(line.startswith(("agent_generation_prompt ", "agent resident trace:",
            "flydelta_model_exchange ", "flydelta_trace ", "flydelta_daemon_arm ")) for line in output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--test-regex", default=DEFAULT_TESTS)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"model file is unavailable: {args.model}")
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, LLAMA_AGENT_MODEL=str(args.model.resolve()),
        LLAMA_AGENT_CHAT_MODEL=str(args.model.resolve()), LLAMA_AGENT_RESIDENT_TRACE="1")
    try:
        devices = subprocess.run([str(args.build_dir / "bin/llama-cli"), "--list-devices"],
            capture_output=True, text=True, env=environment, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        parser.error(f"device discovery unavailable: {error}")
    inventory = devices.stdout + devices.stderr
    (args.output / "devices.log").write_text(inventory)
    nvidia = re.search(r"Vulkan([0-9]+):[^\n]*NVIDIA", inventory)
    if nvidia and not environment.get("GGML_VK_VISIBLE_DEVICES"):
        environment["GGML_VK_VISIBLE_DEVICES"] = nvidia.group(1)
    # -1 uses the existing automatic fitting. With no usable accelerator the
    # native host remains able to run on CPU; do not force full offloading.
    layers = "-1" if re.search(r"Vulkan[0-9]+:", inventory) else "0"
    discovery = subprocess.run(["ctest", "--test-dir", str(args.build_dir), "--show-only=json-v1"],
        capture_output=True, text=True, check=True)
    tests = [test for test in json.loads(discovery.stdout)["tests"]
             if re.search(args.test_regex, test["name"])]
    if not tests:
        parser.error("no registered model tests match the requested profile")
    results = []
    for test in tests:
        command = [*test.get("command", []), "--n-gpu-layers", layers]
        log = args.output / (test["name"] + ".log")
        properties = {item["name"]: item["value"] for item in test.get("properties", [])}
        result = {"name": test["name"], "command": command, "log": str(log),
                  "requested_gpu_layers": layers, "trace_requested": True,
                  "trace_observed": False, "gpu_placement": "unverified"}
        if not command or not pathlib.Path(command[0]).is_file():
            result.update(status="configuration_error", reason="registered executable is missing")
        else:
            print(f"starting {test['name']}: {log}", flush=True)
            with log.open("w") as output:
                process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                    env=environment, cwd=properties.get("WORKING_DIRECTORY", args.build_dir.resolve()))
                sample_path = args.output / (test["name"] + ".gpu.log")
                stop = threading.Event()
                monitor = threading.Thread(target=collect_device_samples,
                    args=(process.pid, stop, sample_path), daemon=True)
                monitor.start()
                try:
                    return_code = process.wait()
                finally:
                    stop.set()
                    monitor.join()
            result["trace_observed"] = trace_observed(log)
            if sample_path.is_file() and sample_path.stat().st_size:
                result["gpu_placement"] = "nvidia_process_observed"
                result["gpu_samples"] = str(sample_path)
            elif layers == "0":
                result["gpu_placement"] = "cpu_requested"
            result.update(exit_code=return_code, status="passed" if return_code == 0 and result["trace_observed"] and
                (layers == "0" or result["gpu_placement"] == "nvidia_process_observed")
                else "incomplete_trace" if return_code == 0 and not result["trace_observed"] else
                "incomplete_gpu_evidence" if return_code == 0 else
                "configuration_error" if return_code == 77 else "failed")
        results.append(result)
        (args.output / "model-profile.json").write_text(json.dumps({"model": str(args.model.resolve()),
            "visible_devices": environment.get("GGML_VK_VISIBLE_DEVICES"),
            "device_evidence": str(args.output / "devices.log"), "tests": results}, indent=2) + "\n")
    return 0 if all(result["status"] == "passed" for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
