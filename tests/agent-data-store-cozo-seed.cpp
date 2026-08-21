#include "agent-data-store-cozo.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

std::vector<std::string> split_csv(const std::string & line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (c == ',' && !quoted) {
            fields.push_back(field);
            field.clear();
        } else {
            field.push_back(c);
        }
    }
    fields.push_back(field);
    return fields;
}

json typed_value(const std::string & value) {
    if (value.empty()) return nullptr;
    bool numeric = true;
    bool decimal = false;
    size_t start = value[0] == '-' ? 1 : 0;
    if (start == value.size()) numeric = false;
    for (size_t i = start; i < value.size(); ++i) {
        if (value[i] == '.' && !decimal) {
            decimal = true;
        } else if (!std::isdigit(static_cast<unsigned char>(value[i]))) {
            numeric = false;
            break;
        }
    }
    if (!numeric) return value;
    try {
        return decimal ? json(std::stod(value)) : json(std::stoll(value));
    } catch (...) {
        return value;
    }
}

bool seed_file(
        common_agent_cozo_data_store & store,
        const std::string & dataset,
        const std::string & path,
        const std::string & id_column,
        std::string & error,
        size_t & count) {
    std::ifstream input(path);
    if (!input) {
        error = "could not open CSV: " + path;
        return false;
    }

    std::string line;
    if (!std::getline(input, line)) {
        error = "CSV is empty: " + path;
        return false;
    }
    auto columns = split_csv(line);
    if (!columns.empty() && columns[0].size() >= 3 &&
            static_cast<unsigned char>(columns[0][0]) == 0xEF &&
            static_cast<unsigned char>(columns[0][1]) == 0xBB &&
            static_cast<unsigned char>(columns[0][2]) == 0xBF) {
        columns[0].erase(0, 3);
    }
    size_t id_index = columns.size();
    for (size_t i = 0; i < columns.size(); ++i) {
        if (columns[i] == id_column || (id_column == "order_id" && columns[i] == "id")) {
            id_index = i;
            break;
        }
    }
    if (id_index == columns.size()) {
        error = "CSV is missing required id column '" + id_column + "': " + path;
        return false;
    }

    const std::string dataset_uri = "dataset://seed/" + dataset;
    count = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split_csv(line);
        json row = json::object();
        for (size_t i = 0; i < columns.size(); ++i) {
            row[columns[i]] = typed_value(i < fields.size() ? fields[i] : std::string());
        }
        const std::string row_id = id_index < fields.size() ? fields[id_index] : std::to_string(count + 1);
        if (row_id.empty() || !store.put_row(dataset_uri, row_id, row.dump(), error)) return false;
        ++count;
    }
    common_agent_dataset_descriptor descriptor;
    descriptor.ref.uri = dataset_uri;
    descriptor.ref.name = dataset;
    descriptor.ref.row_count = count;
    descriptor.ref.column_count = columns.size();
    descriptor.ref.source_resource_uri = "resource://seed/" + dataset;
    descriptor.ref.source_representation = path;
    descriptor.import_processor_id = "smoke.csv-seed";
    descriptor.import_processor_version = "1";
    for (const auto & column : columns) {
        descriptor.columns.push_back({
            column,
            common_agent_dataset_column_type::unknown,
            true});
    }
    if (!store.put_dataset_descriptor(descriptor, error)) return false;
    return true;
}

const char * value(int argc, char ** argv, const char * name) {
    for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == name) return argv[i + 1];
    return nullptr;
}

} // namespace

int main(int argc, char ** argv) {
    const char * db = value(argc, argv, "--db");
    const char * orders = value(argc, argv, "--orders");
    const char * customers = value(argc, argv, "--customers");
    if (db == nullptr || orders == nullptr || customers == nullptr) {
        std::fprintf(stderr, "usage: %s --db PATH --orders PATH --customers PATH\n", argv[0]);
        return 2;
    }

    common_agent_cozo_data_store store;
    std::string error;
    size_t order_count = 0;
    size_t customer_count = 0;
    if (!store.open(db, error) ||
            !seed_file(store, "orders", orders, "order_id", error, order_count) ||
            !seed_file(store, "customers", customers, "customer_id", error, customer_count)) {
        std::fprintf(stderr, "Cozo CSV seed failed: %s\n", error.c_str());
        return 1;
    }
    store.close();
    std::printf("seeded_orders=%zu\nseeded_customers=%zu\n", order_count, customer_count);
    return 0;
}
