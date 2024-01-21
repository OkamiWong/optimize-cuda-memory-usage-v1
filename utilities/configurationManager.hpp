#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "../include/json.hpp"

struct Configuration {
  bool optimize = false;
  bool verify = false;
  int deviceId = 0;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(
    Configuration,
    optimize,
    verify,
    deviceId
  );
};

class ConfigurationManager {
 public:
  static const Configuration& getConfiguration() {
    return configuration;
  };

  static void exportDefaultConfiguration(std::string fileName = "defaultConfig.json") {
    Configuration defaultConfiguration;
    nlohmann::json j = defaultConfiguration;
    std::string s = j.dump(2);

    std::ofstream f(fileName);
    f << s << std::endl;
  }

  static void loadConfiguration(std::string fileName = "config.json") {
    try {
      std::ifstream f(fileName);
      auto j = nlohmann::json::parse(f);
      configuration = j.get<Configuration>();
    } catch (...) {
      std::cerr << "Failed to load configuration from " << fileName << std::endl;
      exit(-1);
    }
  }

 private:
  inline static Configuration configuration;
};
