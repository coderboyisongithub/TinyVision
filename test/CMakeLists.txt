cmake_minimum_required(VERSION 3.10)
project(TinyTorch_test)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(googletest)

add_executable(${PROJECT_NAME}
        main.cpp
        test.cpp
        test_autograd.cpp
        test_function.cpp
        test_module.cpp
        test_optimizer.cpp
        test_tensorimpl.cpp
)
if(${USE_CUDA})
    list(APPEND ${PROJECT_NAME}
            test_cuda_kernel.cpp
            test_cuda_benchmark.cpp
    )
endif ()
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../TinyTorch
        googletest/googletest/include
        googletest/googlemock/include
)
target_link_libraries(${PROJECT_NAME} TinyTorch gtest_main)

include(GoogleTest)

if (${USE_OPENCV})
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS})
    
endif ()

if (${USE_BLAS})
    # copy libopenblas.dll
    if (WIN32)
    find_package(OpenBLAS CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME}  OpenBLAS::OpenBLAS)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/OpenBLAS/win64-64/bin/libopenblas.dll
                $<TARGET_FILE_DIR:${PROJECT_NAME}>/libopenblas.dll
        )
    endif ()
endif ()

# add tests
gtest_discover_tests(${PROJECT_NAME} WORKING_DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME}>)
