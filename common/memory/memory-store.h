#pragma once

#include "memory/memory-types.h"

#include <optional>
#include <string>
#include <vector>

class common_memory_store {
public:
    virtual ~common_memory_store() = default;

    virtual bool open(const std::string & path, std::string & error) = 0;
    virtual void close() = 0;

    virtual bool put(const common_memory_record & record, std::string & error) = 0;
    virtual std::optional<common_memory_record> get(const std::string & id, std::string & error) = 0;
    virtual std::vector<common_memory_record> list(const common_memory_query & query, std::string & error) = 0;
    virtual std::vector<common_memory_hit> search(const common_memory_query & query, std::string & error) = 0;
    virtual bool relate(const std::string & from, const std::string & relation, const std::string & to, float weight, std::string & error) = 0;
    virtual bool erase(const std::string & id, std::string & error) = 0;
};
