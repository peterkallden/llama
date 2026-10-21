#pragma once

#include "agent/data-store.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>
#include <mutex>

// Cozo-backed structured data store. Rows are kept in one host-owned relation
// with JSON payloads so the semantic tool layer is independent of relation
// layout and can later use another backend without changing its contract.
class common_agent_cozo_data_store final : public common_agent_data_store {
public:
    common_agent_cozo_data_store() = default;
    ~common_agent_cozo_data_store() override;

    bool open(const std::string & path, std::string & error);
    void close();
    bool put_row(const std::string & dataset, const std::string & row_id, const std::string & row_json, std::string & error) override;
    bool put_rows(const std::string & dataset, const std::vector<std::pair<std::string, std::string>> & rows, std::string & error) override;
    bool put_dataset_descriptor(const common_agent_dataset_descriptor & descriptor, std::string & error) override;
    bool get_dataset_descriptor(const std::string & dataset_uri, common_agent_dataset_descriptor & descriptor, std::string & error) override;
    bool list_dataset_descriptors(std::vector<common_agent_dataset_descriptor> & descriptors, std::string & error) override;
    bool find_dataset_by_name(const std::string & name, common_agent_dataset_descriptor & descriptor, std::string & error) override;
    bool execute(const std::string & operation, const std::string & request_json, std::string & result_json, std::string & error) override;
    bool execute_scoped(const std::string & operation, const std::string & request_json,
            const common_agent_scope & scope, bool allow_network,
            std::string & result_json, std::string & error) override;
    bool get_investigation_case(const std::string &, const common_agent_scope &,
            common_investigation_case_revision &, std::string &) override;
    bool list_investigation_cases(const common_agent_scope &, size_t,
            const std::string &, std::vector<common_investigation_case_summary> &,
            std::string &, std::string &) override;
    bool publish_investigation_case(const common_investigation_case_revision &,
            uint64_t, const std::string &, common_investigation_case_revision &,
            std::string &) override;
    bool publish_investigation_report(const common_agent_scope &,
            const common_investigation_report_index_entry &,
            common_investigation_report_index_entry &, std::string &) override;
    bool list_investigation_reports(const common_agent_scope &, const std::string &,
            uint64_t, std::vector<common_investigation_report_index_entry> &, std::string &) override;
    bool set_virtual_dataset_source(common_agent_virtual_dataset_source *) override;
    bool authorize_investigation_source(const std::string & source_id,
        const common_agent_scope & scope, std::string & error) const override;

private:
    int32_t db_id_ = -1;
    mutable std::mutex investigation_mutex_;
    common_agent_virtual_dataset_source * virtual_dataset_source_ = nullptr;
    bool run(const std::string & script, const std::string & params_json, std::string & result_json, std::string & error) const;
};
