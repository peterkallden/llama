#include "agent-native-crash.h"

#include <sheredom/subprocess.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <regex>
#include <sstream>
#include <thread>

namespace {

using json = nlohmann::ordered_json;

std::string field(const std::string & text, const char * name) {
    const std::regex pattern(std::string(name) + "=\"([^\"]*)\"");
    std::smatch match;
    return std::regex_search(text, match, pattern) ? match[1].str() : std::string{};
}

std::string trim(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.erase(value.begin());
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
    return value;
}

bool number(const std::string & value, uint32_t & result) {
    if (value.empty()) return false;
    try {
        const auto parsed = std::stoul(value);
        if (parsed > UINT32_MAX) return false;
        result = static_cast<uint32_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

agent_native_crash_frame mi_frame(const std::string & text, size_t fallback_index) {
    agent_native_crash_frame frame;
    const auto level = field(text, "level");
    frame.index = level.empty() ? fallback_index : static_cast<size_t>(std::stoul(level));
    frame.address = field(text, "addr");
    frame.function = field(text, "func");
    frame.module = field(text, "from");
    frame.file = field(text, "file");
    number(field(text, "line"), frame.line);
    return frame;
}

std::string bounded(const std::string & value, size_t limit) {
    return value.size() <= limit ? value : value.substr(0, limit);
}

std::string line_value(const std::string & text, const char * label) {
    const auto start = text.find(label);
    if (start == std::string::npos) return {};
    const auto value_start = start + std::char_traits<char>::length(label);
    const auto value_end = text.find('\n', value_start);
    return trim(text.substr(value_start, value_end == std::string::npos ? std::string::npos : value_end - value_start));
}

std::string first_line_after(const std::string & text, const char * label) {
    const auto start = text.find(label);
    if (start == std::string::npos) return {};
    const auto value_start = text.find('\n', start);
    if (value_start == std::string::npos) return {};
    const auto value_end = text.find('\n', value_start + 1);
    return trim(text.substr(value_start + 1,
        value_end == std::string::npos ? std::string::npos : value_end - value_start - 1));
}

bool resolve_input_path(const std::string & root, const std::string & value, std::string & path, std::string & error) {
    if (root.empty() || value.empty()) { error = "native crash diagnostics requires host-owned paths"; return false; }
    std::error_code fs_error;
    const auto base = std::filesystem::weakly_canonical(root, fs_error);
    if (fs_error) { error = "native crash repository root could not be resolved"; return false; }
    const auto requested = std::filesystem::weakly_canonical(base / value, fs_error);
    if (fs_error) { error = "native crash path could not be resolved"; return false; }
    const auto base_text = base.generic_string();
    const auto requested_text = requested.generic_string();
    if (requested_text != base_text && requested_text.rfind(base_text + "/", 0) != 0) {
        error = "native crash path escapes the host-owned repository";
        return false;
    }
    if (!std::filesystem::is_regular_file(requested, fs_error) || fs_error) {
        error = "native crash executable or dump was not found";
        return false;
    }
    path = requested.string();
    return true;
}

bool run_debugger(
        const std::vector<std::string> & command,
        const agent_native_crash_limits & limits,
        std::string & output,
        int & exit_code,
        bool & timed_out,
        std::string & error) {
    std::vector<char *> argv;
    argv.reserve(command.size() + 1);
    for (const auto & argument : command) argv.push_back(const_cast<char *>(argument.c_str()));
    argv.push_back(nullptr);
    subprocess_s process{};
    if (subprocess_create(argv.data(), subprocess_option_combined_stdout_stderr |
            subprocess_option_enable_async | subprocess_option_no_window |
            subprocess_option_inherit_environment | subprocess_option_search_user_path, &process) != 0) {
        error = "unable to start native crash debugger";
        return false;
    }
    output.clear();
    timed_out = false;
    const auto started = std::chrono::steady_clock::now();
    char buffer[4096];
    while (subprocess_alive(&process)) {
        const auto count = subprocess_read_stdout(&process, buffer, sizeof(buffer));
        if (output.size() < limits.max_output_bytes) output.append(buffer, std::min<size_t>(count, limits.max_output_bytes - output.size()));
        if (limits.timeout_ms != 0 && std::chrono::steady_clock::now() - started >= std::chrono::milliseconds(limits.timeout_ms)) {
            timed_out = true;
            subprocess_terminate(&process);
            break;
        }
        if (count == 0) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    while (const auto count = subprocess_read_stdout(&process, buffer, sizeof(buffer))) {
        if (output.size() < limits.max_output_bytes) output.append(buffer, std::min<size_t>(count, limits.max_output_bytes - output.size()));
    }
    subprocess_join(&process, &exit_code);
    subprocess_destroy(&process);
    error.clear();
    return true;
}

json model_result(const agent_native_crash_result & result) {
    json frames = json::array();
    for (const auto & frame : result.stack) frames.push_back({
        {"index", frame.index}, {"address", frame.address}, {"function", frame.function},
        {"module", frame.module}, {"file", frame.file}, {"line", frame.line},
    });
    json value = {
        {"kind", "native_crash"},
        {"status", agent_native_crash_status_name(result.status)},
        {"summary", result.summary},
        {"symbol_quality", result.symbol_quality},
        {"stack", frames},
    };
    if (!result.signal_or_exception.empty()) value["signal_or_exception"] = result.signal_or_exception;
    if (!result.signal_description.empty()) value["signal_description"] = result.signal_description;
    if (!result.fault_address.empty() || !result.faulting_function.empty() || !result.faulting_module.empty()) {
        value["fault"] = {
            {"address", result.fault_address}, {"function", result.faulting_function}, {"module", result.faulting_module},
        };
    }
    if (!result.failure_bucket.empty()) value["failure_bucket"] = result.failure_bucket;
    return value;
}

} // namespace

std::vector<std::string> agent_gdb_mi_argv(const std::string & executable, const std::string & core_file) {
    return {
        "gdb", "--quiet", "--nx", "--batch", "--interpreter=mi2",
        "-ex", "set pagination off", "-ex", "set confirm off",
        "-ex", "interpreter-exec mi \"-stack-list-frames\"",
        "-ex", "interpreter-exec mi \"-thread-info\"",
        executable, core_file,
    };
}

std::vector<std::string> agent_cdb_argv(const std::string & executable, const std::string & dump_file) {
    return {
        "cdb.exe", "-z", dump_file,
        "-c", "!analyze -v; kv; q", executable,
    };
}

agent_native_crash_result agent_parse_gdb_mi(const std::string & output, const agent_native_crash_limits & limits) {
    agent_native_crash_result result;
    result.backend = "gdb";
    result.raw_output = bounded(output, limits.max_output_bytes);
    result.signal_or_exception = field(output, "signal-name");
    result.signal_description = field(output, "signal-meaning");
    if (result.signal_or_exception.empty()) {
        const std::regex terminated(R"(Program terminated with signal ([^,\s]+),\s*([^\.\n]+))");
        std::smatch match;
        if (std::regex_search(output, match, terminated)) {
            result.signal_or_exception = match[1].str();
            result.signal_description = trim(match[2].str());
        }
    }

    const std::regex frame_pattern(R"(frame=\{([^}]*)\})");
    for (std::sregex_iterator it(output.begin(), output.end(), frame_pattern), end;
            it != end && result.stack.size() < limits.max_frames; ++it) {
        result.stack.push_back(mi_frame((*it)[1].str(), result.stack.size()));
    }
    if (!result.stack.empty()) {
        result.faulting_function = result.stack.front().function;
        result.faulting_module = result.stack.front().module;
        result.fault_address = result.stack.front().address;
        result.symbol_quality = result.stack.front().function.empty() ? "none" : "partial";
        if (!result.stack.front().file.empty()) result.symbol_quality = "full";
    }
    if (result.signal_or_exception.empty() && result.stack.empty()) {
        result.status = agent_native_crash_status::parse_failed;
        result.error = "GDB/MI output did not contain a signal or stack frame";
        return result;
    }
    result.status = agent_native_crash_status::analyzed;
    result.summary = result.signal_or_exception.empty()
        ? "Native crash analyzed by GDB."
        : result.signal_or_exception + " analyzed by GDB.";
    return result;
}

agent_native_crash_result agent_parse_cdb_output(const std::string & output, const agent_native_crash_limits & limits) {
    agent_native_crash_result result;
    result.backend = "cdb";
    result.raw_output = bounded(output, limits.max_output_bytes);
    result.signal_or_exception = line_value(output, "ExceptionCode:");
    result.failure_bucket = line_value(output, "FAILURE_BUCKET_ID:");
    result.faulting_module = line_value(output, "MODULE_NAME:");
    const auto faulting_ip = first_line_after(output, "FAULTING_IP:");
    if (!faulting_ip.empty()) result.fault_address = trim(faulting_ip.substr(0, faulting_ip.find_first_of(" \t")));
    const auto attempted_address = output.find("Attempt to ");
    if (attempted_address != std::string::npos) {
        const auto address = output.find("address ", attempted_address);
        if (address != std::string::npos) result.fault_address = trim(output.substr(address + 8, output.find('\n', address) == std::string::npos ? std::string::npos : output.find('\n', address) - address - 8));
    }

    const auto stack = output.find("STACK_TEXT:");
    if (stack != std::string::npos) {
        std::istringstream lines(output.substr(stack + 12));
        std::string line;
        while (std::getline(lines, line) && result.stack.size() < limits.max_frames) {
            line = trim(line);
            if (line.empty() || line.rfind("FOLLOWUP_", 0) == 0) continue;
            std::istringstream columns(line);
            std::string address, child, return_address, symbol;
            columns >> address >> child >> return_address >> symbol;
            if (symbol.empty() || symbol.find('!') == std::string::npos) continue;
            agent_native_crash_frame frame;
            frame.index = result.stack.size();
            frame.address = address;
            const auto separator = symbol.find('!');
            frame.module = symbol.substr(0, separator);
            frame.function = symbol.substr(separator + 1);
            result.stack.push_back(std::move(frame));
        }
    }
    if (result.signal_or_exception.empty() && result.stack.empty()) {
        result.status = agent_native_crash_status::parse_failed;
        result.error = "CDB output did not contain an exception or stack frame";
        return result;
    }
    result.status = agent_native_crash_status::analyzed;
    result.symbol_quality = result.stack.empty() ? "none" : "partial";
    if (!result.stack.empty()) result.faulting_function = result.stack.front().function;
    if (result.faulting_function.empty() && !result.stack.empty()) result.faulting_function = result.stack.front().module;
    result.summary = result.signal_or_exception.empty()
        ? "Native dump analyzed by CDB."
        : result.signal_or_exception + " analyzed by CDB.";
    return result;
}

const char * agent_native_crash_status_name(agent_native_crash_status status) {
    switch (status) {
    case agent_native_crash_status::analyzed: return "analyzed";
    case agent_native_crash_status::unavailable: return "unavailable";
    case agent_native_crash_status::invalid_input: return "invalid_input";
    case agent_native_crash_status::invalid_dump: return "invalid_dump";
    case agent_native_crash_status::timeout: return "timeout";
    case agent_native_crash_status::process_failed: return "process_failed";
    case agent_native_crash_status::parse_failed: return "parse_failed";
    }
    return "unknown";
}

common_tool_execution_result agent_execute_native_crash(
        const std::string & arguments_json,
        const std::string & backend,
        const std::string & gdb_executable,
        const std::string & cdb_executable,
        const std::string & repository_root,
        const agent_native_crash_limits & limits) {
    const auto args = nlohmann::json::parse(arguments_json, nullptr, false);
    if (!args.is_object() || !args.contains("executable") || !args["executable"].is_string() || !args.contains("dump") || !args["dump"].is_string())
        return common_tool_execution_result::failure("tool.diagnostics.native_crash.invalid_arguments", common_tool_failure_class::validation, false, "Native crash analysis requires executable and dump resources.", "arguments must contain executable and dump");
    std::string executable;
    std::string dump;
    std::string error;
    if (!resolve_input_path(repository_root, args["executable"].get<std::string>(), executable, error) ||
            !resolve_input_path(repository_root, args["dump"].get<std::string>(), dump, error))
        return common_tool_execution_result::failure("tool.diagnostics.native_crash.invalid_dump", common_tool_failure_class::validation, false, "The executable or dump is not available in the host workspace.", std::move(error));

    std::string selected = backend;
    if (selected == "auto") {
#ifdef _WIN32
        selected = "cdb";
#else
        selected = "gdb";
#endif
    }
    std::vector<std::string> command;
    if (selected == "gdb") {
        command = agent_gdb_mi_argv(executable, dump);
        if (!gdb_executable.empty()) command[0] = gdb_executable;
    } else if (selected == "cdb") {
        command = agent_cdb_argv(executable, dump);
        if (!cdb_executable.empty()) command[0] = cdb_executable;
    } else {
        return common_tool_execution_result::failure("tool.diagnostics.native_crash.backend_unavailable", common_tool_failure_class::execution, false, "Native crash analysis is disabled on this host.", "native crash backend is none");
    }

    std::string output;
    int exit_code = 1;
    bool timed_out = false;
    if (!run_debugger(command, limits, output, exit_code, timed_out, error))
        return common_tool_execution_result::failure("tool.diagnostics.native_crash.backend_unavailable", common_tool_failure_class::execution, false, "Native crash debugger could not be started.", std::move(error));
    if (timed_out)
        return common_tool_execution_result::failure("tool.diagnostics.native_crash.timeout", common_tool_failure_class::timeout, true, "Native crash analysis timed out.", bounded(output, limits.max_output_bytes));

    auto result = selected == "gdb" ? agent_parse_gdb_mi(output, limits) : agent_parse_cdb_output(output, limits);
    if (result.status != agent_native_crash_status::analyzed)
        return common_tool_execution_result::failure("tool.diagnostics.native_crash.parse_failed", common_tool_failure_class::execution, false, "The native dump could not be interpreted.", result.error.empty() ? bounded(output, limits.max_output_bytes) : result.error);
    if (exit_code != 0)
        return common_tool_execution_result::failure("tool.diagnostics.native_crash.process_failed", common_tool_failure_class::execution, false, "The native debugger failed while analyzing the dump.", bounded(output, limits.max_output_bytes));
    return common_tool_execution_result::success(model_result(result).dump());
}
