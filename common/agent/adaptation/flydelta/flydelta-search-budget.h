#pragma once

#include "agent/adaptation/flydelta/flydelta-job.h"
#include "agent/adaptation/lifecycle-store.h"

#include <set>

constexpr size_t common_flydelta_orthogonal_control_count = 4;
constexpr size_t common_flydelta_orthogonal_full_generation_top_k = 2;
constexpr size_t common_flydelta_orthogonal_control_rows =
    2 + common_flydelta_orthogonal_control_count + common_flydelta_orthogonal_full_generation_top_k;

struct common_flydelta_search_consumption {
    size_t logical_used = 0;
    size_t physical_rows_started = 0;
    size_t physical_rows_completed = 0;
    size_t physical_rows_unresolved = 0;
};

// These records live in the existing lifecycle journal. A start record is
// written before entering inference; absence of a matching completion means
// physical consumption is unresolved, never zero. Retry limits apply to each
// stable execution group, while logical identities are charged only once.
bool common_flydelta_load_search_consumption(const common_learning_lifecycle_store & store,
        const common_flydelta_experiment_job & job, size_t base_used, size_t limit,
        common_flydelta_search_consumption & consumption, std::string & error);

bool common_flydelta_begin_search_execution(common_learning_lifecycle_store & store,
        const common_flydelta_experiment_job & job, size_t base_used, size_t limit,
        const std::vector<std::string> & logical_ids, const std::string & execution_group,
        size_t max_attempts, std::string & attempt_ref,
        common_flydelta_search_consumption & consumption, std::string & error);

bool common_flydelta_finish_search_execution(common_learning_lifecycle_store & store,
        const common_flydelta_experiment_job & job, const std::string & attempt_ref,
        size_t completed_rows, bool succeeded, std::string & error);
