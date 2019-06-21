project(program)

add_library(static_lib STATIC source/block.c)
add_library(shared_lib SHARED source/block.c)

add_executable(static_block program.c)
target_link_libraries(static_block static_lib)

add_executable(shared_block program.c)
target_link_libraries(shared_block shared_lib)
