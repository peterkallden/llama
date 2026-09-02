#include "astc-vulkan-provenance.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

int main() {
    const unsigned char payload[] = {0, 1, 2, 3, 4, 5};
    astc_vulkan_provenance provenance;
    provenance.source_model = "pythia-test";
    provenance.source_tensor = "blk.0.ffn_down.weight";
    provenance.source_family = "FP16";
    provenance.footprint = "6x6";
    provenance.representation = "gauge";
    provenance.decoder_contract = "scale_l:0.5,scale_a:0.5,offset:-0.25";
    provenance.calibration_hash = "calibration";
    provenance.validation_hash = "validation";
    provenance.holdout_hash = "holdout";
    provenance.selector_config = "gauge-only-v1";
    provenance.validation_prefix = "7";
    provenance.commit_order_hash = "commit-order";
    provenance.padding_contract = "clamp-source-v1";
    provenance.payload_bytes = sizeof(payload);
    provenance.payload_sha256 = astc_vulkan_sha256_hex(payload, sizeof(payload));
    provenance.manifest_sha256 = std::string(64, 'a');
    std::string error;
    assert(astc_vulkan_validate_provenance(provenance, error));

    const std::string path = "astc-vulkan-provenance-test.txt";
    assert(astc_vulkan_write_provenance(path, provenance, error));
    std::ifstream file(path);
    const std::string text((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
    assert(text.find("source_family=FP16\n") != std::string::npos);
    assert(text.find("payload_bytes=6\n") != std::string::npos);
    assert(text.find("payload_sha256=" + provenance.payload_sha256 + "\n") != std::string::npos);
    std::remove(path.c_str());

    provenance.padding_contract = "bad=value";
    assert(!astc_vulkan_validate_provenance(provenance, error));
    std::puts("ASTC Vulkan provenance contract passed");
    return 0;
}
