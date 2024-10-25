#ifndef __CHOREO_OPTIONS_HPP__
#define __CHOREO_OPTIONS_HPP__

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "aux.hpp"

namespace Choreo {

class OptionBase {
public:
  virtual ~OptionBase() {}
  virtual bool Parse(int argc, char** argv, int& currentArg) = 0;
};

template <typename T>
class Option : public OptionBase {
private:
  std::string name;  // option name
  std::string alias; // name alias
  T value;
  T default_value;
  bool requires_arg; // if it requires arguments

public:
  Option(const std::string&, const std::string&, const T&, bool = false);

  bool Parse(int argc, char** argv, int& currentArg) override;

  T GetValue() const { return value; }

  // sugar: conversion and assignment operations
  operator T() const { return value; }
  void operator=(const T& v) { value = v; }
};

class OptionRegistry {
private:
  std::map<std::string, OptionBase*> options;

private:
  // input & output stream
  std::istream* input_stream = nullptr;
  std::ostream* output_stream = nullptr;

  std::string input_filename;
  std::ifstream input_file_stream;
  std::ofstream output_file_stream;

public:
  static OptionRegistry& GetInstance() {
    static OptionRegistry instance;
    return instance;
  }

  void RegisterOption(const std::string& name, OptionBase* option) {
    if (options.count(name)) {
      std::cerr << "option '" << name << "' has been registered twice.\n";
      abort();
    }
    options[name] = option;
  }

  bool Parse(int argc, char** argv) {
    bool stdin_as_input = true;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (options.count(arg)) {
        if (!options[arg]->Parse(argc, argv, i)) return false;
      } else {
        if (!input_filename.empty()) {
          std::cerr << "set input file twice: '" << input_filename << "' and '"
                    << arg << "'.\n";
          return false;
        }
        input_filename = arg;
        stdin_as_input = false;
      }
    }

    if (stdin_as_input)
      assert(input_filename.empty() &&
             "can not set input as both stdin and file.\n");

    return true;
  }

  std::ostream& GetOutputStream() {
    if (output_stream) return *output_stream;
    return std::cout;
  }

  std::istream& GetInputStream() {
    if (input_stream) return *input_stream;

    if (!input_filename.empty()) {
      if (!input_file_stream.is_open()) input_file_stream.open(input_filename);
      input_stream = &input_file_stream;
      return *input_stream;
    }

    return std::cin;
  }

  void SetOutputStream(const std::string& filename) {
    if (!filename.empty()) {
      output_file_stream.open(filename);
      output_stream = &output_file_stream;
    }
  }

  std::string GetInputFileName() { return input_filename; }
};

template <typename T>
inline Option<T>::Option(const std::string& name, const std::string& alias,
                         const T& default_val, bool req)
    : name(name), alias(alias), value(default_val), default_value(default_val),
      requires_arg(req) {
  OptionRegistry::GetInstance().RegisterOption(name, this);
  OptionRegistry::GetInstance().RegisterOption(alias, this);
}

template <typename T>
inline bool Option<T>::Parse(int argc, char** argv, int& currentArg) {
  // be like: -o ab.o
  if (requires_arg) {
    if (currentArg + 1 < argc) {
      std::istringstream iss(argv[++currentArg]);
      iss >> value; // Handle parsing according to type T
      return true;
    } else {
      std::cerr << "Option " << name << " requires an argument." << std::endl;
      return false;
    }
  }

  std::string arg = argv[currentArg];
  auto pos = arg.find('=');
  if (pos != std::string::npos) {
    // be like: -fverbose=true
    std::string name = arg.substr(0, pos);
    std::string valstr = arg.substr(pos + 1);
    std::istringstream iss(valstr);
    iss >> value;
  } else {
    value = default_value;
  }

  return true;
}

// Specialization for boolean type to handle "true" and "false" strings
template <>
bool Option<bool>::Parse(int argc, char** argv, int& currentArg) {
  assert(currentArg < argc &&
         "current argument index exceeds the total count.");
  std::string arg = argv[currentArg];
  auto pos = arg.find('=');
  if (pos != std::string::npos) {
    std::string lowerValue = arg.substr(pos + 1);
    std::transform(lowerValue.begin(), lowerValue.end(), lowerValue.begin(),
                   ::tolower);
    if (lowerValue == "true")
      value = true;
    else if (lowerValue == "false")
      value = false;
    else {
      std::cerr << "Invalid value for boolean option: " << value << std::endl;
      return false;
    }
  } else
    value = true; // set the option on

  return true;
}

} // end namespace Choreo

#endif // __CHOREO_OPTIONS_HPP__
