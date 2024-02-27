#pragma once

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

#include "../include/argh.h"
#include "../include/json.hpp"

struct Configuration {
  // Generic
  bool optimize = false;
  bool verify = false;
  int repeat = 1;

  // Optimization
  bool mergeConcurrentCudaGraphNodes = true;
  double prefetchingBandwidthInGB = 281.0;
  double acceptableRunningTimeFactor = 10.0;
  int minManagedArraySize = 0;

  // Execution
  bool useNvlink = false;
  bool measurePeakMemoryUsage = false;

  // Tiled Cholesky
  int tiledCholeskyN = 256;
  int tiledCholeskyT = 4;

  // LULESH
  int luleshS = 45;
  bool luleshConstrainIterationCount = false;
  int luleshTargetIterationCount = 3000;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(
    Configuration,
    optimize,
    verify,
    repeat,
    mergeConcurrentCudaGraphNodes,
    prefetchingBandwidthInGB,
    acceptableRunningTimeFactor,
    minManagedArraySize,
    useNvlink,
    measurePeakMemoryUsage,
    tiledCholeskyN,
    tiledCholeskyT,
    luleshS,
    luleshConstrainIterationCount,
    luleshTargetIterationCount
  );
};

class ConfigurationManager {
 public:
  static const Configuration& getConfig() {
    assert(initialized);
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
      initialized = true;
    } catch (...) {
      std::cerr << "Failed to load configuration from " << fileName << std::endl;
      std::cerr << "See defaultConfig.json for sample configuration" << std::endl;

      exportDefaultConfiguration();

      exit(-1);
    }
  }

  static void initialize(int argc, char** argv) {
    auto cmdl = argh::parser(argc, argv);
    std::string configFilePath;
    cmdl("configFile", "config.json") >> configFilePath;
    loadConfiguration(configFilePath);
  }

 private:
  inline static Configuration configuration;

  inline static bool initialized = false;

  static void exportConfiguration(const std::string& fileName, const Configuration& configuration) {
    nlohmann::json j = configuration;
    std::string s = j.dump(2);

    std::ofstream f(fileName);
    f << s << std::endl;
  }
};
