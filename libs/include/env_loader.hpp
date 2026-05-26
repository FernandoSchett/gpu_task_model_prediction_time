#ifndef ENV_LOADER_HPP
#define ENV_LOADER_HPP

#include <string>
#include <unordered_map>

using EnvMap = std::unordered_map<std::string, std::string>;

EnvMap load_env_file(const std::string &path);

#endif
