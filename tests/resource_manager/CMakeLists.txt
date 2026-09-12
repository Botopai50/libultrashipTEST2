cmake_minimum_required(VERSION 3.24)
project(ResourceManagerTests LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.3 GIT_SHALLOW TRUE)
FetchContent_Declare(ThreadPool
    GIT_REPOSITORY https://github.com/bshoshany/thread-pool.git
    GIT_TAG v4.1.0 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(spdlog ThreadPool)

get_filename_component(LUS_ROOT "../.." ABSOLUTE)
add_executable(resource_manager_tests
    resource_manager_tests.cpp
    "${LUS_ROOT}/src/ship/resource/ResourceManager.cpp"
    "${LUS_ROOT}/src/ship/resource/Resource.cpp"
    "${LUS_ROOT}/src/ship/utils/Utils.cpp")
target_compile_features(resource_manager_tests PRIVATE cxx_std_20)
target_include_directories(resource_manager_tests PRIVATE
    "${LUS_ROOT}/include" "${threadpool_SOURCE_DIR}/include")
target_link_libraries(resource_manager_tests PRIVATE spdlog::spdlog_header_only)
if(MSVC)
    target_compile_options(resource_manager_tests PRIVATE /utf-8 /Zc:preprocessor)
endif()
enable_testing()
add_test(NAME resource_manager_cache COMMAND resource_manager_tests)
set_tests_properties(resource_manager_cache PROPERTIES TIMEOUT 30)
