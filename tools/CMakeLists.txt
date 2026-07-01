# Project selection is driven by CROQ_PROJECT in the root CMakeLists.txt.

if(CHOREO_BUILD_CO_MOCK)
  add_subdirectory(co-mock)
endif()

if(CHOREO_BUILD_CO_WEB)
  add_subdirectory(co-web)
endif()

if(CHOREO_BUILD_COIR)
  add_subdirectory(coir)
endif()

if(CHOREO_BUILD_CO_PY)
  add_subdirectory(co-py)
endif()
