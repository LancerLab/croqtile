# DepMirror.cmake
# Provides a fallback URL for dependency downloads when the primary URL
# in deps.conf is unreachable.  When the FTP_SERVER environment variable
# is set, builds a mirror URL of the form:
#
#   ftp://${FTP_SERVER}/<category>/<tarball>
#
# Usage:
#   include(cmake/DepMirror.cmake)
#   dep_mirror_fallback(<out_var> <tarball> <category>)
#
# Returns an empty string when FTP_SERVER is unset or empty.

function(dep_mirror_fallback out_var tarball category)
  if(DEFINED ENV{FTP_SERVER} AND NOT "$ENV{FTP_SERVER}" STREQUAL "")
    set(${out_var} "ftp://$ENV{FTP_SERVER}/${category}/${tarball}" PARENT_SCOPE)
  else()
    set(${out_var} "" PARENT_SCOPE)
  endif()
endfunction()
