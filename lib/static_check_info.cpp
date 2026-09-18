#include "static_check_info.hpp"

#include "choreo_template.hpp"
#include "codegen.hpp"
#include "context.hpp"
#include "target.hpp"
#include "types.hpp"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <tuple>
#include <unistd.h>

namespace Choreo {

Option<std::string>
    target_info(OptionKind::User, "-target-info", "--target-info", "",
                "Write target-derived hardware and legality facts as JSON.",
                "-target-info=<file.json>", true);
Option<std::string> static_check_info(
    OptionKind::User, "-static-check-info", "--static-check-info", "",
    "Write compiler static checks and lowering risks as JSON.",
    "-static-check-info=<file.json>", true);
Option<std::string>
    performance_remarks(OptionKind::User, "-Rperformance", "", "",
                        "Emit concise performance remarks for a target group.",
                        "-Rperformance=<group>", true);

Option<bool> static_check_details(
    OptionKind::User, "-fstatic-check-details", "", false,
    "Include per-instance static analysis evidence in the JSON report.");

namespace {
using ErrorKey = std::tuple<std::string, int, int, std::string>;
std::map<ErrorKey, size_t> report_errors;
std::set<std::string> report_dependencies;
std::string preprocessed_hash;
std::string backend_status = "not_run";

std::string JsonEscape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char c : value) {
    switch (c) {
    case '"': out << "\\\""; break;
    case '\\': out << "\\\\"; break;
    case '\b': out << "\\b"; break;
    case '\f': out << "\\f"; break;
    case '\n': out << "\\n"; break;
    case '\r': out << "\\r"; break;
    case '\t': out << "\\t"; break;
    default:
      if (c < 0x20) {
        const char* hex = "0123456789abcdef";
        out << "\\u00" << hex[(c >> 4) & 0xf] << hex[c & 0xf];
      } else {
        out << static_cast<char>(c);
      }
    }
  }
  return out.str();
}

void String(std::ostream& os, const std::string& value) {
  os << '"' << JsonEscape(value) << '"';
}

std::vector<std::string> SupportedFeatureNames(const Target& target,
                                               const ArchId& arch) {
  std::vector<std::string> names;
  for (const auto& feature : target.SupportedFeatures(arch))
    names.push_back(feature.name);
  return names;
}

std::vector<BaseType> ReportTypes(const Target& target, const ArchId& arch) {
  std::set<BaseType> types = target.VectorizableTypes(arch);
  if (target.SupportsExplicitVectorType(arch, BaseType::BOOL, 1))
    types.insert(BaseType::BOOL);
  return {types.begin(), types.end()};
}

uint64_t StableHash(const std::string& value) {
  uint64_t hash = 1469598103934665603ull;
  for (unsigned char c : value) {
    hash ^= c;
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string StableId(const std::string& value) {
  std::ostringstream out;
  out << "fnv1a:" << std::hex << StableHash(value);
  return out.str();
}

std::string FileDigest(const std::string& path) {
  if (path.empty() || path == "-") return "unknown";
  std::ifstream in(path, std::ios::binary);
  if (!in) return "unreadable";
  std::ostringstream contents;
  contents << in.rdbuf();
  return StableId(contents.str());
}

void EmitMemoryTier(std::ostream& os, const Target& target, const ArchId& arch,
                    Storage storage, bool& first) {
  if (storage == Storage::LOCAL && !target.IsLocalStorageSupported(arch))
    return;
  if (storage == Storage::GROUP_SHARED &&
      !target.IsGroupSharedStorageSupported(arch))
    return;

  if (!first) os << ',';
  first = false;
  os << "{\"storage\":";
  String(os, STR(storage));
  os << ",\"capacity_bytes\":" << target.GetMemCapacity(storage, arch);
  if (storage != Storage::GLOBAL)
    os << ",\"alignment_bytes\":" << target.GetMemAlignmentByte(storage, arch);
  else
    os << ",\"alignment_bytes\":0";
  os << '}';
}

} // namespace

void EmitStaticCheckContextJson(std::ostream& os) {
  const auto& registry = OptionRegistry::GetInstance();
  const auto input = registry.GetInputFileName();
  os << "\"input\":{";
  os << "\"path\":";
  String(os, input.empty() ? "<stdin>" : input);
  os << ",\"content_hash\":";
  String(os, FileDigest(input));
  os << "},\"options\":{";
  os << "\"invocation_hash\":";
  String(os, StableId(registry.GetInvocation()));
  os << ",\"effective\":{";
  bool first_option = true;
  for (const auto& [name, value] : registry.EffectiveOptions()) {
    if (name == "-static-check-info" || name == "-target-info" ||
        name == "-fstatic-check-details" || name == "-Rperformance")
      continue;
    if (!first_option) os << ',';
    first_option = false;
    String(os, name);
    os << ':';
    String(os, value);
  }
  os << "}},\"environment\":{";
  bool first_env = true;
  auto environment = CCtx().GetTarget().ReportEnvironmentVariables();
  environment.insert(environment.begin(), "EXTRA_TARGET_CFLAGS");
  for (const auto* name : {"CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH"})
    environment.emplace_back(name);
  for (const auto& name : environment) {
    if (!first_env) os << ',';
    first_env = false;
    String(os, name);
    os << ':';
    if (const char* value = std::getenv(name.c_str()))
      String(os, value);
    else
      os << "null";
  }
  os << "},\"dependencies\":{\"coverage\":\"preprocessor_observed; "
        "backend_sdk_not_enumerated\",\"files\":[";
  bool first = true;
  for (const auto& dependency : report_dependencies) {
    if (!first) os << ',';
    first = false;
    os << "{\"path\":";
    String(os, dependency);
    os << ",\"content_hash\":";
    String(os, FileDigest(dependency));
    os << '}';
  }
  os << "]},\"preprocessed_hash\":";
  if (preprocessed_hash.empty())
    os << "null";
  else
    String(os, preprocessed_hash);
  os << ",\"compiler\":{";
#ifdef CHOREO_SDK_VERSION
  os << "\"version\":";
  String(os, CHOREO_SDK_VERSION);
#else
  os << "\"version\":\"unknown\"";
#endif
#ifdef CHOREO_GIT_REVISION
  os << ",\"base_revision\":";
  String(os, CHOREO_GIT_REVISION);
#endif
  os << ",\"binary_hash\":";
  static const auto compiler_hash = FileDigest("/proc/self/exe");
  String(os, compiler_hash);
  os << "}";
}

void RecordStaticCheckDependency(const std::string& path) {
  if (!static_check_info.GetValue().empty()) report_dependencies.insert(path);
}

void RecordStaticCheckPreprocessedInput(const std::string& input) {
  if (!static_check_info.GetValue().empty())
    preprocessed_hash = StableId(input);
}

void RecordBackendCompilation(bool succeeded) {
  backend_status = succeeded ? "completed" : "failed";
}

void EmitBackendFactsJson(std::ostream& os) {
  os << "{\"compilation\":";
  String(os, backend_status);
  os << ",\"resources\":{\"status\":";
  String(os, backend_status == "not_run" ? "not_run" : "unavailable");
  os << ",\"code\":\"BACKEND_RESOURCE_METADATA_NOT_COLLECTED\","
        "\"registers\":null,\"spill_bytes\":null}}";
}

void EmitTargetInfoJson(std::ostream& os) {
  const auto& target = CCtx().GetTarget();
  const auto arch = CCtx().GetArch();
  const auto features = SupportedFeatureNames(target, arch);
  const auto types = ReportTypes(target, arch);

  os << "{\n  \"schema_version\":1,\n  \"kind\":\"target-info\",\n";
  os << "  \"compiler\":{";
#ifdef CHOREO_SDK_VERSION
  os << "\"version\":";
  String(os, CHOREO_SDK_VERSION);
#else
  os << "\"version\":\"unknown\"";
#endif
#ifdef CHOREO_GIT_REVISION
  os << ",\"revision\":";
  String(os, CHOREO_GIT_REVISION);
#else
  os << ",\"revision\":\"unknown\"";
#endif
  os << "},\n";
  os << "  \"target\":{\"name\":";
  String(os, target.Name());
  os << ",\"device_name\":";
  String(os, target.DeviceName());
  os << ",\"arch\":";
  String(os, arch);
  os << "},\n";

  os << "  \"features\":[";
  for (size_t i = 0; i < features.size(); ++i) {
    if (i) os << ',';
    String(os, features[i]);
  }
  os << "],\n";

  os << "  \"limits\":{\n"
     << "    \"vector_register_bytes\":" << target.GetVectorLength(arch)
     << ",\n"
     << "    \"vectorize_limit\":" << target.VectorizeLimit(arch) << ",\n"
     << "    \"explicit_vector\":"
     << (target.SupportsExplicitVector(arch) ? "true" : "false") << ",\n"
     << "    \"local_mem_budget_bytes\":" << target.GetLocalMemBudget(arch)
     << ",\n"
     << "    \"max_threads_per_block\":" << target.GetMaxThreadsPerBlock(arch)
     << ",\n"
     << "    \"max_threads_per_sm\":" << target.GetMaxThreadsPerSM(arch)
     << ",\n"
     << "    \"max_parallel_by_count\":{";
  bool first_level = true;
  for (auto level : target.GetParallelLevels(arch)) {
    auto max_count = target.GetMaxParallelByCount(level, arch);
    if (!max_count) continue;
    if (!first_level) os << ',';
    first_level = false;
    String(os, STR(level));
    os << ':' << max_count;
  }
  os << "}\n  },\n";

  os << "  \"memory_tiers\":[";
  bool first_memory = true;
  EmitMemoryTier(os, target, arch, Storage::LOCAL, first_memory);
  EmitMemoryTier(os, target, arch, Storage::GROUP_SHARED, first_memory);
  EmitMemoryTier(os, target, arch, Storage::SHARED, first_memory);
  EmitMemoryTier(os, target, arch, Storage::GLOBAL, first_memory);
  os << "],\n";

  os << "  \"vector_layouts\":[";
  bool first_type = true;
  for (auto type : types) {
    const auto element_bytes = type == BaseType::BOOL ? 4 : SizeOf(type);
    const auto physical_lanes =
        element_bytes ? target.GetVectorLength(arch) / element_bytes : 0;
    if (!first_type) os << ',';
    first_type = false;
    os << "{\"element_type\":";
    String(os, STR(type));
    os << ",\"element_bytes\":" << element_bytes
       << ",\"physical_lanes\":" << physical_lanes
       << ",\"legal_lane_ranges\":[[1," << physical_lanes << ']';
    if (physical_lanes) {
      os << ",[" << 2 * physical_lanes << ',' << 2 * physical_lanes << "],["
         << 4 * physical_lanes << ',' << 4 * physical_lanes << ']';
    }
    os << "] ,\"probe\":{";
    os << "\"one\":"
       << (target.SupportsExplicitVectorType(arch, type, 1) ? "true" : "false");
    if (physical_lanes) {
      os << ",\"physical\":"
         << (target.SupportsExplicitVectorType(arch, type, physical_lanes)
                 ? "true"
                 : "false")
         << ",\"double\":"
         << (target.SupportsExplicitVectorType(arch, type, 2 * physical_lanes)
                 ? "true"
                 : "false")
         << ",\"quad\":"
         << (target.SupportsExplicitVectorType(arch, type, 4 * physical_lanes)
                 ? "true"
                 : "false");
    }
    os << "}}";
  }
  os << "],\n";

  os << "  \"semantics\":{\n"
     << "    \"event\":" << (target.IsEventSupported() ? "true" : "false")
     << ",\n"
     << "    \"mma\":" << (target.IsMMASupported() ? "true" : "false") << ",\n"
     << "    \"async_dma\":"
     << (target.IsFeatureSupported(arch, STR(ChoreoFeature::ASYNC_DMA))
             ? "true"
             : "false")
     << ",\n"
     << "    \"directional_fence\":"
     << (target.SupportsDirectionalFence(arch) ? "true" : "false") << ",\n"
     << "    \"seq_cst_fence\":"
     << (target.SupportsSeqCstFence(arch) ? "true" : "false") << "\n"
     << "  }\n}\n";
}

void EmitAssessmentFactsJson(std::ostream& os, bool finalized) {
  auto kind = [](UsageType type) {
    switch (type) {
    case UsageType::ElementAccess: return "element_bounds";
    case UsageType::ShapeCompatibility: return "shape_compatibility";
    case UsageType::LoopBound: return "loop_bounds";
    case UsageType::HardwareConstraint: return "hardware_constraint";
    default: return "other";
    }
  };
  std::map<std::string, std::map<std::string, size_t>> counts;
  std::map<std::string, size_t> checks;
  std::map<std::string, std::set<std::string>> check_functions;
  for (const auto& [function, context] : CCtx().GetAllFunctionContexts()) {
    const auto& assessor = context.GetAssessor();
    const auto& assertions = assessor.GetAssertions();
    auto display = function;
    if (const auto* info = FindChoreoTemplateInstance(function))
      display = info->display_name;
    for (const auto& check : assessor.GetAssessmentLog()) {
      std::string status;
      const Assertion* assertion = check.assertion_idx < assertions.size()
                                       ? &assertions[check.assertion_idx]
                                       : nullptr;
      bool runtime_enabled = assertion && assertion->enabled;
      if (assertion && assertion->duplicate_of < assertions.size())
        runtime_enabled = assertions[assertion->duplicate_of].enabled;
      switch (check.outcome) {
      case AssessOutcome::STATIC_TRUE: status = "proved"; break;
      case AssessOutcome::STATIC_FALSE: status = "violated"; break;
      case AssessOutcome::UNKNOWN: status = "unproven"; break;
      case AssessOutcome::RUNTIME:
        status = !finalized        ? "runtime_required"
                 : runtime_enabled ? "runtime_checked"
                                   : "runtime_disabled";
        break;
      }
      ++counts[kind(check.usage_type)][status];
      // Detailed proof records stay on demand. Failed/disabled checks are
      // visible in the compact report, with an explicit truncation count.
      if (!static_check_details.GetValue() &&
          (status == "proved" || status == "runtime_checked"))
        continue;
      std::ostringstream item;
      item << "{\"kind\":";
      String(item, kind(check.usage_type));
      item << ",\"status\":";
      String(item, status);
      item << ",\"source\":{\"path\":";
      String(item, check.loc.begin.get_filename());
      item << ",\"line\":" << check.loc.begin.get_line()
           << ",\"column\":" << check.loc.begin.get_column()
           << "},\"predicate\":";
      String(item, UnScopedExpr(check.predicate));
      item << ",\"guard\":";
      String(item, UnScopedExpr(check.guard));
      item << ",\"message\":";
      String(item, UnScopedExpr(check.message));
      if (assertion)
        item << ",\"duplicate\":" << (assertion->duplicate ? "true" : "false");
      item << '}';
      ++checks[item.str()];
      check_functions[item.str()].insert(display);
    }
  }
  os << "{\"scope\":\"recorded_semantic_checks_under_source_assumptions\","
        "\"counts\":{";
  bool first = true;
  for (const auto& [category, statuses] : counts) {
    if (!first) os << ',';
    first = false;
    String(os, category);
    os << ":{";
    bool first_status = true;
    for (const auto& [status, count] : statuses) {
      if (!first_status) os << ',';
      first_status = false;
      String(os, status);
      os << ':' << count;
    }
    os << '}';
  }
  os << "},\"checks\":[";
  size_t emitted = 0;
  for (const auto& [item, count] : checks) {
    if (!static_check_details.GetValue() && emitted == 20) break;
    if (emitted++) os << ',';
    os << item.substr(0, item.size() - 1) << ",\"occurrences\":" << count
       << ",\"functions\":[";
    bool first_function = true;
    for (const auto& function : check_functions.at(item)) {
      if (!first_function) os << ',';
      first_function = false;
      String(os, function);
    }
    os << "]}";
  }
  os << "],\"omitted_check_groups\":" << checks.size() - emitted << '}';
}

void EmitMemoryFactsJson(std::ostream& os) {
  os << "{\"measurement\":\"compiler_static_peak_live_bytes\",\"functions\":[";
  bool first = true;
  const auto& stats = CCtx().GetMemUsageStats();
  for (const auto& [function, tiers] : stats.functions) {
    if (!first) os << ',';
    first = false;
    os << "{\"function\":";
    const auto* info = FindChoreoTemplateInstance(function);
    String(os, info ? info->display_name : function);
    os << ",\"tiers\":[";
    bool first_tier = true;
    for (const auto& [storage, tier] : tiers) {
      if (!first_tier) os << ',';
      first_tier = false;
      os << "{\"storage\":";
      String(os, STR(storage));
      os << ",\"peak_bytes\":" << tier.peak_bytes << ",\"peak_bytes_kind\":";
      auto runtime_tiers = stats.runtime_decided_functions.find(function);
      bool runtime_pool =
          runtime_tiers != stats.runtime_decided_functions.end() &&
          runtime_tiers->second.count(storage);
      String(os, runtime_pool              ? "runtime_pool_reservation"
                 : tier.has_runtime_extent ? "static_lower_bound"
                                           : "static_peak");
      os << ",\"limit_bytes\":" << tier.limit_bytes
         << ",\"has_runtime_extent\":"
         << (tier.has_runtime_extent ? "true" : "false") << '}';
    }
    os << "]}";
  }
  os << "]}";
}

void RecordStaticCheckError(const std::string& file, int line, int column,
                            const std::string& message) {
  if (!static_check_info.GetValue().empty())
    ++report_errors[{file, line, column, message}];
}

void EmitIncompleteStaticCheckInfo(std::ostream& os, const std::string& status,
                                   const std::string& stage, int exit_code) {
  os << "{\n  \"schema_version\":3,\n  \"kind\":\"static-check-info\",\n"
     << "  \"context\":{";
  EmitStaticCheckContextJson(os);
  os << "},\n  \"compilation\":{\"status\":";
  String(os, status);
  os << ",\"stage\":";
  String(os, stage);
  os << ",\"exit_code\":" << exit_code << "},\n"
     << "  \"analysis\":\"not_run\",\n  \"compiler_diagnostics\":[";
  bool first = true;
  size_t emitted = 0;
  for (const auto& [key, count] : report_errors) {
    if (!static_check_details.GetValue() && emitted == 3) break;
    ++emitted;
    const auto& [file, line, column, message] = key;
    if (!first) os << ',';
    first = false;
    os << "{\"source\":{\"path\":";
    String(os, file);
    os << ",\"line\":" << line << ",\"column\":" << column << "},\"message\":";
    String(os, message);
    os << ",\"occurrences\":" << count << '}';
  }
  os << "],\n  \"omitted_diagnostic_groups\":" << report_errors.size() - emitted
     << ",\n  \"assessments\":";
  EmitAssessmentFactsJson(os);
  os << ",\n  \"backend\":";
  EmitBackendFactsJson(os);
  os << "\n}\n";
}

void EmitGenericStaticCheckInfo(std::ostream& os) {
  os << "{\n  \"schema_version\":3,\n  \"kind\":\"static-check-info\",\n"
     << "  \"context\":{";
  EmitStaticCheckContextJson(os);
  os << "},\n  \"compilation\":{\"status\":\"completed\",\"exit_code\":0},\n"
     << "  \"analysis\":\"unsupported_target\"\n}\n";
}

bool WriteReportFile(const std::string& path,
                     const std::function<void(std::ostream&)>& emit) {
  if (path.empty()) return false;
  if (path == "-") {
    emit(std::cout);
    return static_cast<bool>(std::cout);
  }
  // Publish complete JSON atomically, in the destination filesystem.
  std::string temporary = path + ".XXXXXX";
  int fd = mkstemp(temporary.data());
  if (fd < 0) {
    errs() << "error: cannot open report file '" << path << "'.\n";
    return false;
  }
  close(fd);
  std::ofstream out(temporary);
  emit(out);
  out.close();
  if (!out || std::rename(temporary.c_str(), path.c_str()) != 0) {
    std::remove(temporary.c_str());
    errs() << "error: failed while writing report file '" << path << "'.\n";
    return false;
  }
  return true;
}

} // namespace Choreo
