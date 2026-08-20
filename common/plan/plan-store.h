#pragma once
#include "plan/plan-types.h"
#include <optional>
class common_plan_store { public: virtual ~common_plan_store() = default; virtual bool open(const std::string & path, std::string & error) = 0; virtual void close() = 0; virtual bool create(const common_plan_state & plan, std::string & error) = 0; virtual std::optional<common_plan_state> get(const std::string & plan_id, std::string & error) = 0; virtual std::vector<common_plan_state> list(std::string & error) = 0; virtual bool apply(const common_plan_operation & operation, common_plan_state & updated_plan, std::string & error) = 0; virtual std::vector<common_plan_event> history(const std::string & plan_id, std::string & error) = 0; virtual bool erase(const std::string & plan_id, std::string & error) = 0; };
