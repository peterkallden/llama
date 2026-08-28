#pragma once

#include "agent/adaptation/adaptation-observer.h"
#include "agent/adaptation/learning-observation.h"

#include <filesystem>
#include <string>
#include <vector>

struct common_learning_transaction {
    int schema_version = 1;
    std::string id;
    common_learning_observation observation;
    std::string created_at;
};

bool common_learning_transaction_validate(
        const common_learning_transaction & transaction,
        size_t max_evidence,
        std::string & error);

std::string common_learning_transaction_to_json(const common_learning_transaction & transaction);
bool common_learning_transaction_from_json(
        const std::string & text,
        common_learning_transaction & transaction,
        std::string & error);

class common_learning_transaction_store {
public:
    virtual ~common_learning_transaction_store() = default;
    virtual bool append(const common_learning_transaction & transaction, std::string & error) = 0;
    virtual bool contains_idempotency(const std::string & key, bool & contains, std::string & error) const = 0;
    virtual std::vector<common_learning_transaction> list(std::string & error) const = 0;
};

class common_learning_in_memory_transaction_store final : public common_learning_transaction_store {
public:
    bool append(const common_learning_transaction & transaction, std::string & error) override;
    bool contains_idempotency(const std::string & key, bool & contains, std::string & error) const override;
    std::vector<common_learning_transaction> list(std::string & error) const override;

private:
    std::vector<common_learning_transaction> transactions;
};

class common_learning_jsonl_transaction_store final : public common_learning_transaction_store {
public:
    bool open(const std::filesystem::path & path, std::string & error);
    bool append(const common_learning_transaction & transaction, std::string & error) override;
    bool contains_idempotency(const std::string & key, bool & contains, std::string & error) const override;
    std::vector<common_learning_transaction> list(std::string & error) const override;

private:
    std::filesystem::path path;
};

struct common_learning_transaction_observer_config {
    bool collection_allowed = false;
    size_t max_evidence = 16;
};

class common_learning_transaction_observer final : public common_agent_adaptation_observer {
public:
    common_learning_transaction_observer(
            common_learning_transaction_store & store,
            common_learning_transaction_observer_config config = {});

    bool observe(
            const common_agent_request & request,
            const common_plan_state & plan,
            const common_agent_result & result,
            std::string & error) override;

private:
    common_learning_transaction_store & store;
    common_learning_transaction_observer_config config;
};
