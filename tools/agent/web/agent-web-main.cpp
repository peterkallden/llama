#include "agent-web-server.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --daemon-port N [--listen ADDRESS] [--port N] "
        "[--daemon-address ADDRESS] [--web-bearer-token TOKEN | --web-bearer-token-env ENV] "
        "[--daemon-authorization VALUE | --daemon-authorization-env ENV] "
        "[--allowed-origin ORIGIN]\n",
        argv0);
}

bool value_arg(int & index, int argc, char ** argv, std::string & value) {
    if (++index >= argc) return false;
    value = argv[index];
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    agent_web_server_options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        std::string value;
        if (arg == "--listen" && value_arg(i, argc, argv, value)) options.listen_address = value;
        else if (arg == "--port" && value_arg(i, argc, argv, value)) options.port = std::stoi(value);
        else if (arg == "--daemon-address" && value_arg(i, argc, argv, value)) options.daemon_address = value;
        else if (arg == "--daemon-port" && value_arg(i, argc, argv, value)) options.daemon_port = std::stoi(value);
        else if (arg == "--web-bearer-token" && value_arg(i, argc, argv, value)) options.web_bearer_token = value;
        else if (arg == "--web-bearer-token-env" && value_arg(i, argc, argv, value)) {
            const char * token = std::getenv(value.c_str());
            if (token == nullptr || token[0] == '\0') {
                std::fprintf(stderr, "web bearer token environment variable is empty: %s\n", value.c_str());
                return 2;
            }
            options.web_bearer_token = token;
        }
        else if (arg == "--daemon-authorization" && value_arg(i, argc, argv, value)) options.daemon_authorization = value;
        else if (arg == "--daemon-authorization-env" && value_arg(i, argc, argv, value)) {
            const char * authorization = std::getenv(value.c_str());
            if (authorization == nullptr || authorization[0] == '\0') {
                std::fprintf(stderr, "daemon authorization environment variable is empty: %s\n", value.c_str());
                return 2;
            }
            options.daemon_authorization = authorization;
        }
        else if (arg == "--allowed-origin" && value_arg(i, argc, argv, value)) options.allowed_origin = value;
        else {
            usage(argv[0]);
            return 2;
        }
    }
    std::string error;
    if (!run_agent_web_server(options, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    return 0;
}
