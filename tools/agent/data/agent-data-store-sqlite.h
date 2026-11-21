#pragma once

#include "agent/data-store.h"
#include "agent/storage/sqlite/sqlite-database.h"
#include <mutex>

class common_agent_sqlite_data_store final : public common_agent_data_store {
public:
    ~common_agent_sqlite_data_store() override;

    bool open(const std::string & path, std::string & error);
    void close();
    bool put_row(const std::string & dataset, const std::string & row_id, const std::string & row_json, std::string & error) override;
    bool put_rows(const std::string & dataset, const std::vector<std::pair<std::string, std::string>> & rows, std::string & error) override;
    bool put_dataset_descriptor(const common_agent_dataset_descriptor & descriptor, std::string & error) override;
    bool get_dataset_descriptor(const std::string & dataset_uri, common_agent_dataset_descriptor & descriptor, std::string & error) override;
    bool list_dataset_descriptors(std::vector<common_agent_dataset_descriptor> & descriptors, std::string & error) override;
    bool find_dataset_by_name(const std::string & name, common_agent_dataset_descriptor & descriptor, std::string & error) override;
    bool execute(const std::string & operation, const std::string & request_json, std::string & result_json, std::string & error) override;
    bool get_research_revision(const std::string &, uint64_t, const common_agent_scope &, common_agent_research_revision_ref &, std::string &) override;
    bool publish_research_revision(const common_agent_scope &, const common_agent_research_revision_ref &, uint64_t, common_agent_research_revision_ref &, bool &, std::string &) override;
    bool get_research_draft(const std::string &, const common_agent_scope &, common_agent_research_draft_ref &, std::string &) override;
    bool save_research_draft(const common_agent_scope &, const common_agent_research_draft_ref &, uint64_t, common_agent_research_draft_ref &, bool &, std::string &) override;

private:
    bool ensure_schema(std::string & error);
    common_sqlite_database database_;
    common_sqlite_database research_database_;
    std::mutex research_mutex_;
};
