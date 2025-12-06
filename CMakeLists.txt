cmake_minimum_required(VERSION 3.20)
project(TinyGPT VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mcpu=native -ffast-math")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")
    endif()
endif()

# The one model that works: word-level language model
add_executable(tiny_lm src/tiny_lm.cpp)
target_link_libraries(tiny_lm PRIVATE m)

# Install
install(TARGETS tiny_lm DESTINATION bin)

