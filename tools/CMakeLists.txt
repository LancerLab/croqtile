option(CHOREO_BUILD_CO_MOCK "Build the mock interpreter tool (co-mock)" ON)

if(CHOREO_BUILD_CO_MOCK)
  add_subdirectory(co-mock)
endif()
