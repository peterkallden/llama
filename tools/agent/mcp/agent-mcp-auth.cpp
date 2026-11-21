#include "agent-mcp-auth.h"

#include "common/http.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>

#if defined(CPPHTTPLIB_OPENSSL_SUPPORT)
#include <openssl/bn.h>
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/rsa.h>
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
#include <openssl/core_names.h>
#include <openssl/param_build.h>
#endif
#endif

namespace {

std::string base64url_decode(const std::string & value, bool & ok) {
    std::string input = value;
    for (char & c : input) {
        if (c == '-') c = '+';
        if (c == '_') c = '/';
    }
    while (input.size() % 4 != 0) input.push_back('=');
    std::string output;
    output.reserve((input.size() * 3) / 4);
    static const std::string alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    uint32_t buffer = 0;
    int bits = 0;
    ok = true;
    for (const char c : input) {
        if (c == '=') break;
        const auto position = alphabet.find(c);
        if (position == std::string::npos) {
            ok = false;
            return {};
        }
        buffer = (buffer << 6) | static_cast<uint32_t>(position);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            output.push_back(static_cast<char>((buffer >> bits) & 0xff));
        }
    }
    return output;
}

bool contains_string(const std::vector<std::string> & values, const std::string & value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

bool audience_matches(const nlohmann::ordered_json & claims, const std::string & expected) {
    if (claims["aud"].is_string()) return claims["aud"].get<std::string>() == expected;
    if (!claims["aud"].is_array()) return false;
    for (const auto & entry : claims["aud"]) {
        if (entry.is_string() && entry.get<std::string>() == expected) return true;
    }
    return false;
}

bool scopes_match(const nlohmann::ordered_json & claims, const std::vector<std::string> & required) {
    if (required.empty()) return true;
    std::vector<std::string> scopes;
    if (claims.contains("scope") && claims["scope"].is_string()) {
        std::istringstream input(claims["scope"].get<std::string>());
        std::string scope;
        while (input >> scope) scopes.push_back(scope);
    }
    if (claims.contains("scp") && claims["scp"].is_array()) {
        for (const auto & scope : claims["scp"]) {
            if (scope.is_string()) scopes.push_back(scope.get<std::string>());
        }
    }
    for (const auto & required_scope : required) {
        if (!contains_string(scopes, required_scope)) return false;
    }
    return true;
}

bool valid_time_claim(
        const nlohmann::ordered_json & claims,
        const char * name,
        int64_t now,
        uint32_t skew,
        bool required,
        bool future) {
    if (!claims.contains(name)) return !required;
    if (!claims[name].is_number()) return false;
    const int64_t value = claims[name].get<int64_t>();
    return future ? value <= now + static_cast<int64_t>(skew)
                  : value + static_cast<int64_t>(skew) >= now;
}

#if defined(CPPHTTPLIB_OPENSSL_SUPPORT)
EVP_PKEY * make_rsa_public_key(
        const std::string & modulus,
        const std::string & exponent) {
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
    BIGNUM * n = BN_bin2bn(
        reinterpret_cast<const unsigned char *>(modulus.data()),
        static_cast<int>(modulus.size()), nullptr);
    BIGNUM * e = BN_bin2bn(
        reinterpret_cast<const unsigned char *>(exponent.data()),
        static_cast<int>(exponent.size()), nullptr);
    OSSL_PARAM_BLD * builder = OSSL_PARAM_BLD_new();
    EVP_PKEY_CTX * context = EVP_PKEY_CTX_new_from_name(nullptr, "RSA", nullptr);
    OSSL_PARAM * parameters = nullptr;
    EVP_PKEY * key = nullptr;

    if (n != nullptr && e != nullptr && builder != nullptr &&
            OSSL_PARAM_BLD_push_BN(builder, OSSL_PKEY_PARAM_RSA_N, n) == 1 &&
            OSSL_PARAM_BLD_push_BN(builder, OSSL_PKEY_PARAM_RSA_E, e) == 1) {
        parameters = OSSL_PARAM_BLD_to_param(builder);
        if (parameters != nullptr && context != nullptr &&
                EVP_PKEY_fromdata_init(context) == 1 &&
                EVP_PKEY_fromdata(context, &key, EVP_PKEY_PUBLIC_KEY, parameters) != 1) {
            EVP_PKEY_free(key);
            key = nullptr;
        }
    }

    OSSL_PARAM_free(parameters);
    EVP_PKEY_CTX_free(context);
    OSSL_PARAM_BLD_free(builder);
    BN_free(n);
    BN_free(e);
    return key;
#else
    RSA * rsa = RSA_new();
    BIGNUM * n = BN_bin2bn(
        reinterpret_cast<const unsigned char *>(modulus.data()),
        static_cast<int>(modulus.size()), nullptr);
    BIGNUM * e = BN_bin2bn(
        reinterpret_cast<const unsigned char *>(exponent.data()),
        static_cast<int>(exponent.size()), nullptr);
    if (rsa == nullptr || n == nullptr || e == nullptr ||
            RSA_set0_key(rsa, n, e, nullptr) != 1) {
        BN_free(n);
        BN_free(e);
        RSA_free(rsa);
        return nullptr;
    }
    EVP_PKEY * key = EVP_PKEY_new();
    if (key == nullptr || EVP_PKEY_assign_RSA(key, rsa) != 1) {
        EVP_PKEY_free(key);
        RSA_free(rsa);
        return nullptr;
    }
    return key;
#endif
}
#endif

} // namespace

bool agent_mcp_opaque_token_authenticator::register_token(
        std::string token,
        agent_mcp_caller_policy policy,
        std::string & error) {
    if (token.empty()) {
        error = "MCP auth token must not be empty";
        return false;
    }
    if (policy.caller_id.empty() || policy.audience.empty() ||
            policy.namespace_id.empty() || policy.tool_profile.empty()) {
        error = "MCP caller policy requires caller_id, audience, namespace_id and tool_profile";
        return false;
    }
    for (const auto & entry : entries_) {
        if (entry.token == token || entry.policy.caller_id == policy.caller_id) {
            error = "duplicate MCP auth token or caller_id";
            return false;
        }
    }
    entries_.push_back({std::move(token), std::move(policy)});
    error.clear();
    return true;
}

bool agent_mcp_opaque_token_authenticator::authenticate(
        const agent_mcp_authentication_request & request,
        agent_mcp_caller_policy & policy,
        std::string & error) const {
    constexpr const char * prefix = "Bearer ";
    if (request.authorization.rfind(prefix, 0) != 0 ||
            request.authorization.size() <= std::char_traits<char>::length(prefix)) {
        error = "Bearer authentication is required";
        return false;
    }
    const std::string token = request.authorization.substr(std::char_traits<char>::length(prefix));
    for (const auto & entry : entries_) {
        if (entry.token == token) {
            policy = entry.policy;
            error.clear();
            return true;
        }
    }
    error = "Bearer token is not authorized";
    return false;
}

bool agent_mcp_policy_allows_tool(
        const agent_mcp_caller_policy & policy,
        const std::string & tool_name) {
    return policy.allowed_tools.empty() ||
        std::find(policy.allowed_tools.begin(), policy.allowed_tools.end(), tool_name) != policy.allowed_tools.end();
}

bool agent_mcp_policy_allows_tool(
        const agent_mcp_caller_policy & policy,
        const std::string & tool_name,
        bool read_only,
        bool requires_confirmation) {
    if (!agent_mcp_policy_allows_tool(policy, tool_name)) {
        return false;
    }
    if (policy.allow_writes) {
        return true;
    }
    // A caller without write authority may only see and invoke read-only
    // tools. Confirmation metadata does not grant authority by itself.
    return read_only && !requires_confirmation;
}

agent_mcp_jwt_authenticator::agent_mcp_jwt_authenticator(
        agent_mcp_jwt_authenticator_options options)
    : options_(std::move(options)) {
}

bool agent_mcp_jwt_authenticator::refresh_jwks(std::string & error) const {
    try {
        auto [client, url] = common_http_client(options_.jwks_uri);
        client.set_connection_timeout(5);
        client.set_read_timeout(5);
        const auto response = client.Get(url.path);
        if (!response || response->status != 200) {
            error = "JWT JWKS request failed";
            return false;
        }
        auto parsed = nlohmann::ordered_json::parse(response->body, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object() ||
                !parsed.contains("keys") || !parsed["keys"].is_array()) {
            error = "JWT JWKS response is invalid";
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        jwks_ = std::move(parsed);
        jwks_loaded_at_ = std::chrono::steady_clock::now();
        error.clear();
        return true;
    } catch (const std::exception & exception) {
        error = std::string("JWT JWKS request failed: ") + exception.what();
        return false;
    }
}

bool agent_mcp_jwt_authenticator::verify_signature(
        const nlohmann::ordered_json & header,
        const std::string & signing_input,
        const std::string & signature,
        std::string & error) const {
    const std::string algorithm = header.value("alg", "");
    if (!contains_string(options_.allowed_algorithms, algorithm) || algorithm == "none") {
        error = "JWT algorithm is not allowed";
        return false;
    }
#if !defined(CPPHTTPLIB_OPENSSL_SUPPORT)
    error = "JWT verification requires an OpenSSL-enabled build";
    return false;
#else
    const std::string key_id = header.value("kid", "");
    std::lock_guard<std::mutex> lock(mutex_);
    const auto keys = jwks_.value("keys", nlohmann::ordered_json::array());
    const nlohmann::ordered_json * selected = nullptr;
    for (const auto & key : keys) {
        if (key.is_object() && key.value("kid", "") == key_id &&
                key.value("kty", "") == "RSA") {
            selected = &key;
            break;
        }
    }
    if (selected == nullptr) {
        error = "JWT signing key was not found in JWKS";
        return false;
    }
    bool ok = false;
    const auto modulus = base64url_decode(selected->value("n", ""), ok);
    if (!ok) {
        error = "JWT RSA modulus is invalid";
        return false;
    }
    const auto exponent = base64url_decode(selected->value("e", ""), ok);
    if (!ok) {
        error = "JWT RSA exponent is invalid";
        return false;
    }
    const auto signature_bytes = base64url_decode(signature, ok);
    if (!ok) {
        error = "JWT signature encoding is invalid";
        return false;
    }
    EVP_PKEY * key = make_rsa_public_key(modulus, exponent);
    if (key == nullptr) {
        error = "JWT EVP key construction failed";
        return false;
    }
    EVP_MD_CTX * context = EVP_MD_CTX_new();
    const bool verified = context != nullptr &&
        EVP_DigestVerifyInit(context, nullptr, EVP_sha256(), nullptr, key) == 1 &&
        EVP_DigestVerifyUpdate(context, signing_input.data(), signing_input.size()) == 1 &&
        EVP_DigestVerifyFinal(
            context,
            reinterpret_cast<const unsigned char *>(signature_bytes.data()),
            signature_bytes.size()) == 1;
    if (context != nullptr) EVP_MD_CTX_free(context);
    EVP_PKEY_free(key);
    if (!verified) {
        error = "JWT signature verification failed";
        return false;
    }
    error.clear();
    return true;
#endif
}

bool agent_mcp_jwt_authenticator::authenticate(
        const agent_mcp_authentication_request & request,
        agent_mcp_caller_policy & policy,
        std::string & error) const {
    constexpr const char * prefix = "Bearer ";
    if (request.authorization.rfind(prefix, 0) != 0) {
        error = "Bearer authentication is required";
        return false;
    }
    const std::string token = request.authorization.substr(std::char_traits<char>::length(prefix));
    const auto first_dot = token.find('.');
    const auto second_dot = token.find('.', first_dot == std::string::npos ? first_dot : first_dot + 1);
    if (first_dot == std::string::npos || second_dot == std::string::npos ||
            token.find('.', second_dot + 1) != std::string::npos) {
        error = "JWT must contain three segments";
        return false;
    }
    bool ok = false;
    const auto header_text = base64url_decode(token.substr(0, first_dot), ok);
    if (!ok) { error = "JWT header encoding is invalid"; return false; }
    const auto claims_text = base64url_decode(token.substr(first_dot + 1, second_dot - first_dot - 1), ok);
    if (!ok) { error = "JWT claims encoding is invalid"; return false; }
    const auto header = nlohmann::ordered_json::parse(header_text, nullptr, false);
    const auto claims = nlohmann::ordered_json::parse(claims_text, nullptr, false);
    if (header.is_discarded() || claims.is_discarded() ||
            !header.is_object() || !claims.is_object()) {
        error = "JWT header or claims are invalid";
        return false;
    }
    const int64_t now = static_cast<int64_t>(std::time(nullptr));
    if (claims.value("iss", "") != options_.issuer ||
            !audience_matches(claims, options_.audience) ||
            !valid_time_claim(claims, "exp", now, options_.clock_skew_seconds, true, false) ||
            !valid_time_claim(claims, "nbf", now, options_.clock_skew_seconds, false, true) ||
            !scopes_match(claims, options_.required_scopes)) {
        error = "JWT claims do not satisfy the configured issuer, audience, time or scope policy";
        return false;
    }
    bool refresh = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto age = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - jwks_loaded_at_).count();
        refresh = jwks_.is_null() || age >= static_cast<int64_t>(options_.jwks_cache_seconds);
    }
    if (refresh && !refresh_jwks(error)) return false;
    if (!verify_signature(
            header,
            token.substr(0, second_dot),
            token.substr(second_dot + 1),
            error)) {
        return false;
    }
    policy = options_.policy_template;
    policy.caller_id = claims.value(options_.subject_claim, "");
    if (policy.caller_id.empty()) {
        error = "JWT subject claim is missing";
        return false;
    }
    error.clear();
    return true;
}
