set(HETERO_TARGET_SOURCES
  hetero_codegen.cpp
  hetero_target.cpp
)

add_library(HeteroTarget OBJECT ${HETERO_TARGET_SOURCES})

target_include_directories(HeteroTarget
  PUBLIC include
  PRIVATE ${CMAKE_SOURCE_DIR}
  PRIVATE ${CMAKE_SOURCE_DIR}/lib/Target/CC
)
