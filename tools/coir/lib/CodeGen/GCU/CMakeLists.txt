# ---------- Kurama toolchain (auto-download, like LLVMBootstrap) ----------
set(_KURAMA_ROOT "${CMAKE_SOURCE_DIR}/extern/kurama")
set(_KURAMA_MARKER "${_KURAMA_ROOT}/bin/gcu-compiler-opt")

set(_DEP_CONF "${CMAKE_CURRENT_SOURCE_DIR}/kurama-dep.conf")
if(EXISTS "${_DEP_CONF}")
  file(STRINGS "${_DEP_CONF}" _kur_lines REGEX "^[A-Z_0-9]+=")
  foreach(_line ${_kur_lines})
    string(REGEX MATCH "^([A-Z_0-9]+)=(.*)" _ "${_line}")
    set(_KUR_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
  endforeach()
endif()

option(COIR_AUTO_DOWNLOAD_KURAMA
  "Automatically download the kurama toolchain if not found" ON)

if(NOT EXISTS "${_KURAMA_MARKER}" AND COIR_AUTO_DOWNLOAD_KURAMA AND _KUR_KURAMA_URL)
  set(_KUR_TAR "${CMAKE_SOURCE_DIR}/extern/${_KUR_KURAMA_TAR}")

  if(NOT EXISTS "${_KUR_TAR}")
    message(STATUS "GCU native: kurama not found, downloading...")
    file(DOWNLOAD "${_KUR_KURAMA_URL}" "${_KUR_TAR}"
      SHOW_PROGRESS
      EXPECTED_MD5 "${_KUR_KURAMA_MD5}"
      STATUS _kur_dl_status
      TIMEOUT 300)
    list(GET _kur_dl_status 0 _kur_dl_code)
    if(NOT _kur_dl_code EQUAL 0)
      list(GET _kur_dl_status 1 _kur_dl_msg)
      file(REMOVE "${_KUR_TAR}")
      message(WARNING
        "Failed to download kurama toolchain: ${_kur_dl_msg}\n"
        "GCU native codegen will be disabled.\n"
        "Run: make -C extern setup-kurama-kit FTP_SERVER=172.16.11.18")
    endif()
  endif()

  if(EXISTS "${_KUR_TAR}")
    message(STATUS "GCU native: extracting kurama into ${_KURAMA_ROOT}...")
    file(MAKE_DIRECTORY "${_KURAMA_ROOT}")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xzf "${_KUR_TAR}"
      WORKING_DIRECTORY "${_KURAMA_ROOT}"
      RESULT_VARIABLE _kur_extract_result)
    if(NOT _kur_extract_result EQUAL 0)
      message(WARNING "Failed to extract kurama tarball: ${_KUR_TAR}")
    else()
      file(REMOVE "${_KUR_TAR}")
      message(STATUS "GCU native: kurama installed to ${_KURAMA_ROOT}")
    endif()
  endif()
endif()

# ---- Library target ----
add_library(MLIRCoIRCodeGenGCU STATIC
  EmitTopscc.cpp
  ConvertToGCU.cpp
  GCUCodeGen.cpp
)
target_include_directories(MLIRCoIRCodeGenGCU PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
  ${CMAKE_CURRENT_BINARY_DIR}/../../../include
)
target_link_libraries(MLIRCoIRCodeGenGCU PRIVATE
  MLIRCoIRCodeGenCommon
  MLIRCoIRDialect
  MLIRArithDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRPass
  MLIRIR
  MLIRSupport
  LLVMSupport
)

if(EXISTS "${_KURAMA_MARKER}")
  target_compile_definitions(MLIRCoIRCodeGenGCU PRIVATE
    __CHOREO_KURAMA_DIR__="${_KURAMA_ROOT}"
  )
endif()
