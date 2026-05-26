option(CHOREO_BUILD_CO_MOCK "Build the mock interpreter tool (co-mock)" ON)
option(CHOREO_BUILD_CO_WEB "Build the web playground (co-web, requires Emscripten)" ON)

if(CHOREO_BUILD_CO_MOCK)
  add_subdirectory(co-mock)
endif()

if(CHOREO_BUILD_CO_WEB)
  add_subdirectory(co-web)
endif()
