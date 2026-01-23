#ifndef __CHOREO_OPTIONS_HPP__
#define __CHOREO_OPTIONS_HPP__

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

#include "aux.hpp"
#include "target_registry.hpp"

namespace Choreo {

// forward declarations
std::ostream& errs();
std::ostream& dbgs();

enum class OptionKind {
  User = 0,
  Hidden = 1,
  Internal = 2,
};

class OptionBase {
public:
  OptionKind kind = OptionKind::User;
  OptionBase(OptionKind ok) : kind(ok) {}
  virtual ~OptionBase() {}
  virtual bool Parse(int argc, char** argv, int& currentArg) = 0;
  virtual const std::string Description() const = 0;
  virtual const std::string GetName() const = 0;
  virtual const std::string GetAlias() const = 0;

public:
  virtual void SetError(const std::string&) = 0;
  virtual const std::string GetError() const = 0;
};

template <typename T>
class Option : public OptionBase {
private:
  std::string name;  // option name
  std::string alias; // name alias
  T value;
  T default_value;
  std::string description;   // explanation of this option
  std::string option_desc;   // for describing the option, if needed
  bool requires_arg = false; // if it requires extra argument

public:
  Option(OptionKind, const std::string&, const std::string&, const T&,
         const std::string& = "", const std::string& = "", bool = false);
  ~Option();

  bool Parse(int argc, char** argv, int& currentArg) override;

  T GetValue() const { return value; }

  // sugar: conversion and assignment operations
  operator T() const { return value; }
  void operator=(const T& v) { value = v; }

  const std::string Description() const override;

  const std::string GetName() const override { return name; }
  const std::string GetAlias() const override { return alias; }

private:
  std::string err;

public:
  void SetError(const std::string& e) override { err = e; }
  const std::string GetError() const override { return err; }
};

class OptionRegistry {
private:
  std::unordered_map<std::string, OptionBase*> options;

private:
  // input & output stream
  std::istream* input_stream = nullptr;
  std::ostream* output_stream = nullptr;

  std::string input_filename;
  std::string output_filename;
  std::ifstream input_file_stream;
  std::ofstream output_file_stream;

  bool stdout_as_output = false;
  bool stdin_as_input = false;

  std::ostringstream ess;
  int ret_code = 0;

public:
  const std::string GetOutputFileName() const { return output_filename; }
  const std::string GetInputFileName() const { return input_filename; }
  const std::string GetInputName() const {
    return RemoveDirectoryPrefix(RemoveSuffix(input_filename, ".co"));
  }

  bool StdoutAsOutput() const { return stdout_as_output; }
  bool StdinAsInput() const { return stdin_as_input; }

  static OptionRegistry& GetInstance() {
    static OptionRegistry instance;
    return instance;
  }

  void Reset() {
    input_stream = nullptr;
    output_stream = nullptr;

    input_filename.clear();
    output_filename.clear();
    input_file_stream.close();
    output_file_stream.close();

    stdout_as_output = false;
    stdin_as_input = false;

    ess.str("");
    ret_code = 0;
  }

  void RegisterOption(const std::string& name, OptionBase* option) {
    if (options.count(name)) {
      errs() << "option '" << name << "' has been registered twice.\n";
      abort();
    }
    options[name] = option;
  }

  void UnRegisterOption(const std::string& name) {
    if (options.count(name)) {
      auto* option = options.at(name);
      if (option->GetAlias() != "") {
        auto alias = option->GetAlias();
        options.erase(alias);
      }
      options.erase(name);
    }
  }

  bool Parse(int argc, char** argv) {
    Reset();
    for (int i = 1; i < argc; ++i)
      if (!Parse(argc, argv, i)) return false;

    if (!stdin_as_input && input_filename.empty()) {
      ess << "error: no input file.";
      ret_code = 1;
      return false;
    }
    return true;
  }

  bool Parse(int argc, char** argv, int& i) {
    assert(i < argc && "the argument is out of bound.");
    (void)argc;

    std::string arg = argv[i];
    auto option = arg;
    if (auto pos = option.find("="); pos != std::string::npos)
      option = arg.substr(0, pos);
    if (option == "--help" || option == "-H") {
      Help(OptionKind::User);
      return false;
    } else if (option == "--help-target") {
      std::cout << "The supported compile targets including: ";
      for (auto& ti : TargetRegistry::List())
        std::cout << "\n - " << ti.name << ": " << ti.description;
      std::cout << "\n";
      return false;
    } else if (option == "--help-hidden") {
      Help(OptionKind::Hidden);
      return false;
    }
    if (options.count(option)) {
      if (!options[option]->Parse(argc, argv, i)) {
        ess << options[option]->GetError();
        return false;
      }
    } else {
      if (!input_filename.empty()) {
        ess << "error: set input file twice: '" << input_filename << "' and '"
            << arg << "'.";
        ret_code = 1;
        return false;
      } else
        input_filename = arg;
      if (input_filename == "-") stdin_as_input = true;
    }

    return true;
  }

  const std::string Message() { return ess.str(); }
  int ReturnCode() { return ret_code; }

  std::ostream& GetOutputStream() {
    if (output_stream) return *output_stream;
    stdout_as_output = true;
    return std::cout; // directly output to stdout
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
    if (!filename.empty() && filename != "-") {
      output_filename = filename;
      output_file_stream.open(filename);
      output_stream = &output_file_stream;
    }
  }

  std::string GetInputFileName() { return input_filename; }

public:
  void Help(OptionKind ok) {
    std::cout << "Usage: choreo [options] file...\n";
    std::cout << "Options:\n";
    std::cout << "  " << std::setw(26) << std::left << "--help"
              << "Display this information.\n";
    std::cout << "  " << std::setw(26) << std::left << "--help-hidden"
              << "Display hidden options.\n";

    // apply
    std::vector<std::string> keys;
    for (auto& opt_item : options) keys.push_back(opt_item.first);
    std::sort(keys.begin(), keys.end());

    std::set<OptionBase*> visited;
    for (auto& name : keys) {
      auto* option = options.at(name);

      // since alias name is also registered, avoid the duplicated printing
      if (visited.count(option)) continue;
      visited.insert(option);

      if ((int)option->kind <= (int)ok) {
        auto desc = option->Description();
        if (!desc.empty()) std::cout << desc << "\n";
      }
    }
    std::cout << "\n";
    ret_code = 0;
  }
};

template <typename T>
inline const std::string Option<T>::Description() const {
  std::ostringstream oss;

  std::string opt_desc;
  if (!alias.empty())
    opt_desc = alias + ", " + ((option_desc.empty()) ? name : option_desc);
  else
    opt_desc = (option_desc.empty()) ? name : option_desc;
  oss << "  " << std::setw(26) << std::left << opt_desc;

  // add a new line if it is lengthy
  if (opt_desc.size() < 26)
    oss << description;
  else
    oss << "\n                            " << description;

  return oss.str();
}

template <typename T>
inline Option<T>::Option(OptionKind ok, const std::string& name,
                         const std::string& alias, const T& default_val,
                         const std::string& desc, const std::string& opt_d,
                         bool req)
    : OptionBase(ok), name(name), alias(alias), value(default_val),
      default_value(default_val), description(desc), option_desc(opt_d),
      requires_arg(req) {
  OptionRegistry::GetInstance().RegisterOption(name, this);
  if (!alias.empty()) OptionRegistry::GetInstance().RegisterOption(alias, this);
}

template <typename T>
inline Option<T>::~Option() {
  OptionRegistry::GetInstance().UnRegisterOption(name);
}

template <typename T>
inline bool Option<T>::Parse(int argc, char** argv, int& currentArg) {
  // be like: -o ab.o, requires an extra parameter
  if (requires_arg) {
    std::string arg = argv[currentArg];
    auto pos = arg.find('=');
    if (pos != std::string::npos) {
      std::string valstr = arg.substr(pos + 1);
      std::istringstream iss(valstr);
      iss >> value;
      return true;
    }
    if (currentArg + 1 < argc) {
      std::istringstream iss(argv[++currentArg]);
      iss >> value; // Handle parsing according to type T
      return true;
    }
    SetError("Option " + name + " requires an argument.");
    return false;
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

  // support ';' seperated options
  if constexpr (std::is_same_v<T, std::string>)
    std::replace(value.begin(), value.end(), ';', ' ');

  return true;
}

// Specialization for boolean type to handle "true" and "false" strings
template <>
inline bool Option<bool>::Parse(int argc, char** argv, int& currentArg) {
  assert(currentArg < argc &&
         "current argument index exceeds the total count.");
  (void)argc;

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
      std::ostringstream es;
      es << "Invalid value for boolean option: " << value << ".";
      SetError(es.str());
      return false;
    }
  } else
    value = true; // set the option on

  return true;
}

} // end namespace Choreo

#endif // __CHOREO_OPTIONS_HPP__
