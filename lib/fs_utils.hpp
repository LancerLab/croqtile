#ifndef __CHOREO_FS_UTILS_HPP__
#define __CHOREO_FS_UTILS_HPP__

#include <filesystem>
#include <string>

namespace Choreo {

inline std::string GetAbsPath(const std::filesystem::path& cwd,
                              const std::string& relative_path) {
  std::filesystem::path rel_path(relative_path);
  auto abs_path = std::filesystem::weakly_canonical(cwd / rel_path);
  return abs_path.parent_path().string();
}

} // namespace Choreo

#endif // __CHOREO_FS_UTILS_HPP__
