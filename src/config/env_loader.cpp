#include "env_loader.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <string>

namespace {

std::string trim(const std::string &input) {
    auto first = input.begin();
    while (first != input.end() && std::isspace(static_cast<unsigned char>(*first))) {
        ++first;
    }

    auto last = input.end();
    while (last != first && std::isspace(static_cast<unsigned char>(*(last - 1)))) {
        --last;
    }

    return std::string(first, last);
}

std::string strip_optional_quotes(const std::string &input) {
    if (input.size() >= 2) {
        const char first = input.front();
        const char last = input.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            return input.substr(1, input.size() - 2);
        }
    }
    return input;
}

}  // namespace

EnvMap load_env_file(const std::string &path) {
    EnvMap values;
    std::ifstream file(path);
    if (!file) {
        return values;
    }

    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        const std::string export_prefix = "export ";
        if (line.rfind(export_prefix, 0) == 0) {
            line = trim(line.substr(export_prefix.size()));
        }

        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }

        const std::string key = trim(line.substr(0, eq));
        const std::string value = strip_optional_quotes(trim(line.substr(eq + 1)));
        if (!key.empty()) {
            values[key] = value;
        }
    }

    return values;
}
