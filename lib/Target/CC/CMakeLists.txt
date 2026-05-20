set(CC_TARGET_SOURCES
  cc_codegen.cpp
  cc_target.cpp
)

add_library(CCTarget OBJECT ${CC_TARGET_SOURCES})

target_include_directories(CCTarget
  PUBLIC include
  PRIVATE ${CMAKE_SOURCE_DIR}
)
