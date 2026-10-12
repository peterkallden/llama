#pragma once

#include "agent/adaptation/flydelta/flydelta-evaluator.h"
#include "agent/adaptation/flydelta/flydelta-representation-diagnostics.h"
#include "agent/adaptation/flydelta/flydelta-scale-search.h"

#include <iosfwd>

void print_margin(
        std::ostream & output,
        const char * prefix,
        const common_flydelta_decision_margin & margin,
        const common_flydelta_decision_margin * baseline = nullptr);

void print_representation_diagnostics(
        std::ostream & output,
        const common_flydelta_representation_diagnostics & diagnostics);

void print_representation_diagnostics(
        std::ostream & output,
        const common_flydelta_scale_geometry & geometry);
