#include "pocs/agent/smoke/support/synthesis-review-scenario.h"
#include <iostream>
int main() {
    for(const bool rejected:{false,true}) {
        const auto result=synthesis_review_scenario::run(rejected);
        for(const auto & event:result.trace) std::cout<<"trace: "<<event.detail<<'\n';
        std::cout<<"review outcome="<<(rejected?"unverified_draft":"accepted_partial")<<'\n';
    }
}
