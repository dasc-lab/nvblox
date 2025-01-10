add_executable(eval
    src/eval.cpp
)
target_include_directories(eval PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(eval
    nvblox_lib nvblox_datasets
)
