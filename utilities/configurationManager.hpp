#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "../include/json.hpp"

struct Configuration {
  double prefetchingBandwidthInGB = 281.0;
  double acceptableRunningTimeFactor = 10.0;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(
    Configuration,
    prefetchingBandwidthInGB,
    acceptableRunningTimeFactor
  );
};

class ConfigurationManager {
 public:
  static const Configuration& getConfig() {
    return configuration;
  };

  static void exportDefaultConfiguration(std::string fileName = "defaultConfig.json") {
    Configuration defaultConfiguration;
    exportConfiguration(fileName, defaultConfiguration);
  }

  static void exportCurrentConfiguration(std::string fileName = "currentConfig.json") {
    exportConfiguration(fileName, configuration);
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

  static void exportConfiguration(const std::string& fileName, const Configuration& configuration) {
    nlohmann::json j = configuration;
    std::string s = j.dump(2);

    std::ofstream f(fileName);
    f << s << std::endl;
  }
};
