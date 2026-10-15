#pragma once

#include <string>

// Replace invalid UTF-8 bytes with U+FFFD before text is serialized as JSON.
inline std::string agent_resource_normalize_utf8(const std::string & input) {
    std::string output;
    output.reserve(input.size());
    for (size_t index = 0; index < input.size();) {
        const auto first = static_cast<unsigned char>(input[index]);
        size_t length = 0;
        if (first <= 0x7f) length = 1;
        else if (first >= 0xc2 && first <= 0xdf) length = 2;
        else if (first >= 0xe0 && first <= 0xef) length = 3;
        else if (first >= 0xf0 && first <= 0xf4) length = 4;

        bool valid = length != 0 && index + length <= input.size();
        if (valid && length > 1) {
            const auto second = static_cast<unsigned char>(input[index + 1]);
            valid = (second & 0xc0) == 0x80 &&
                !(first == 0xe0 && second < 0xa0) &&
                !(first == 0xed && second >= 0xa0) &&
                !(first == 0xf0 && second < 0x90) &&
                !(first == 0xf4 && second > 0x8f);
            for (size_t offset = 2; valid && offset < length; ++offset) {
                valid = (static_cast<unsigned char>(input[index + offset]) & 0xc0) == 0x80;
            }
        }
        if (valid) {
            output.append(input, index, length);
            index += length;
        } else {
            output.append("\xef\xbf\xbd");
            ++index;
        }
    }
    return output;
}
