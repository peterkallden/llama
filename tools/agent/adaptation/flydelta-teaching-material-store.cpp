#include "flydelta-teaching-material-store.h"

#ifdef LLAMA_AGENT_ADAPTATION_USE_COZO
extern "C" {
#include <cozo_c.h>
}
#include <nlohmann/json.hpp>
#endif

#include <memory>
#include <cstdint>

#ifdef LLAMA_AGENT_ADAPTATION_USE_COZO
namespace {

using json = nlohmann::ordered_json;

class cozo_teaching_material_backend {
public:
    ~cozo_teaching_material_backend() { close(); }

    bool open(const std::string & path, std::string & error) {
        if (path.empty()) {
            error = "Cozo teaching-material store requires a path";
            return false;
        }
        char * open_error = cozo_open_db("sqlite", path.c_str(), "{}", &db_id_);
        if (open_error) {
            error = open_error;
            cozo_free_str(open_error);
            db_id_ = -1;
            return false;
        }
        const char * schema = R"COZO(
            {
                ?[store_id, payload_json] <- [['__probe__', '{}']]
                :create agent_flydelta_teaching_material {
                    store_id: String =>
                    payload_json: String
                }
            }
            {
                ?[store_id] <- [['__probe__']]
                :delete agent_flydelta_teaching_material { store_id }
            }
        )COZO";
        std::string relations;
        if (!run("::relations", "{}", relations, error)) {
            close();
            return false;
        }
        const auto parsed = json::parse(relations, nullptr, false);
        bool present = false;
        if (parsed.is_object() && parsed.contains("rows") && parsed["rows"].is_array()) {
            for (const auto & row : parsed["rows"]) {
                if (row.is_array() && !row.empty() && row[0].is_string() &&
                        row[0].get<std::string>() == "agent_flydelta_teaching_material") {
                    present = true;
                    break;
                }
            }
        }
        if (!present) {
            std::string ignored;
            if (!run(schema, "{}", ignored, error)) {
                close();
                return false;
            }
        }
        return true;
    }

    bool load(const std::string & key, std::string & snapshot, std::string & error) const {
        const json params = {{"store_id", key}};
        std::string result;
        if (!run("?[payload_json] := *agent_flydelta_teaching_material[store_id, payload_json], store_id == $store_id",
                params.dump(), result, error)) return false;
        const auto value = json::parse(result, nullptr, false);
        if (!value.is_object() || !value.contains("rows") || !value["rows"].is_array()) {
            error = "Cozo teaching-material query returned invalid rows";
            return false;
        }
        if (value["rows"].empty()) {
            snapshot.clear();
            return true;
        }
        if (!value["rows"][0].is_array() || value["rows"][0].empty() ||
                !value["rows"][0][0].is_string()) {
            error = "Cozo teaching-material payload row is invalid";
            return false;
        }
        snapshot = value["rows"][0][0].get<std::string>();
        return true;
    }

    bool save(const std::string & key, const std::string & snapshot, std::string & error) const {
        const json params = {{"store_id", key}, {"payload_json", snapshot}};
        std::string result;
        return run(
            "?[store_id, payload_json] <- [[$store_id, $payload_json]] :put agent_flydelta_teaching_material { store_id => payload_json }",
            params.dump(), result, error);
    }

private:
    bool run(const std::string & script, const std::string & params,
            std::string & result, std::string & error) const {
        if (db_id_ < 0) {
            error = "Cozo teaching-material store is not open";
            return false;
        }
        char * raw = cozo_run_query(db_id_, script.c_str(), params.c_str(), false);
        if (!raw) {
            error = "Cozo teaching-material query failed without diagnostic output";
            return false;
        }
        result = raw;
        cozo_free_str(raw);
        const auto parsed = json::parse(result, nullptr, false);
        if (parsed.is_object() && parsed.value("ok", true) == false) {
            error = parsed.value("message", std::string("Cozo teaching-material query failed"));
            return false;
        }
        return true;
    }

    void close() {
        if (db_id_ >= 0) {
            cozo_close_db(db_id_);
            db_id_ = -1;
        }
    }

    int32_t db_id_ = -1;
};

} // namespace
#endif

std::shared_ptr<common_flydelta_teaching_material_runtime>
make_agent_flydelta_teaching_material_runtime(
        const std::string & requested_backend,
        const std::string & path,
        common_flydelta_teaching_material_identity identity,
        std::string & error) {
    error.clear();
    if (!common_flydelta_teaching_material_identity_validate(identity, error)) return {};

    std::string backend = requested_backend.empty() ? "auto" : requested_backend;
    if (backend == "auto") backend = path.empty() ? "in-memory" : "cozo";
    if (backend == "in-memory") {
        if (!path.empty()) {
            error = "in-memory teaching-material backend does not accept a path";
            return {};
        }
        return std::make_shared<common_flydelta_teaching_material_runtime>(std::move(identity));
    }
    if (backend != "cozo") {
        error = "unsupported teaching-material backend: " + backend;
        return {};
    }
#ifdef LLAMA_AGENT_ADAPTATION_USE_COZO
    auto backend_store = std::make_shared<cozo_teaching_material_backend>();
    if (!backend_store->open(path, error)) return {};
    const std::string store_key = common_flydelta_teaching_material_compatibility_key(identity);
    if (store_key.empty()) {
        error = "teaching-material compatibility key is empty";
        return {};
    }
    common_flydelta_teaching_material_persistence persistence;
    persistence.load = [backend_store, store_key](std::string & snapshot, std::string & load_error) {
        return backend_store->load(store_key, snapshot, load_error);
    };
    persistence.save = [backend_store, store_key](const std::string & snapshot, std::string & save_error) {
        return backend_store->save(store_key, snapshot, save_error);
    };
    auto runtime = std::make_shared<common_flydelta_teaching_material_runtime>(
        std::move(identity), std::move(persistence));
    if (!runtime->load(error)) return {};
    return runtime;
#else
    error = "teaching-material Cozo backend requires a build with LLAMA_MEMORY_COZO";
    return {};
#endif
}
