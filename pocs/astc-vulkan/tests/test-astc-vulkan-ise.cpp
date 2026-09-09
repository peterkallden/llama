#include "astc-vulkan-bit-budget.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

void roundtrip(astc_vulkan_ise_range range, const std::vector<uint16_t> & input) {
    const auto require = [](bool condition) {
        if (!condition) std::abort();
    };
    std::vector<uint8_t> packed;
    std::vector<uint16_t> unpacked;
    require(astc_vulkan_ise_pack(range, input, packed));
    const auto layout = astc_vulkan_ise_evaluate(range, input.size());
    require(layout.valid && packed.size() == (layout.total_bits + 7u) / 8u);
    require(astc_vulkan_ise_unpack(range, input.size(), packed, unpacked));
    require(unpacked == input);
}

} // namespace

int main() {
    const auto require = [](bool condition) {
        if (!condition) std::abort();
    };
    roundtrip(astc_vulkan_ise_choose_range(1, 16),
              {0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1});
    roundtrip(astc_vulkan_ise_choose_range(2, 5), {0, 1, 2, 2, 1});
    roundtrip(astc_vulkan_ise_choose_range(4, 3), {4, 0, 2});
    roundtrip(astc_vulkan_ise_choose_range(2, 7), {0, 2, 1, 2, 0, 1, 2});
    roundtrip(astc_vulkan_ise_choose_range(4, 5), {4, 3, 2, 1, 0});

    const auto legal = astc_vulkan_evaluate_bit_budget({32, 4, 40, 15, 3, 128});
    require(legal.legal && legal.used_bits == 128 && legal.remaining_bits == 0);
    const auto legal_choices = astc_vulkan_enumerate_bit_budget({32, 4, 40, 15, 3, 128});
    require(!legal_choices.empty());
    for (const auto & choice : legal_choices) {
        require(choice.legal && choice.used_bits <= 128u);
    }
    const auto rejected = astc_vulkan_evaluate_bit_budget({32, 4, 40, 31, 4, 128});
    require(!rejected.legal && rejected.used_bits > rejected.fixed_block_bits + 96u);
    return 0;
}
