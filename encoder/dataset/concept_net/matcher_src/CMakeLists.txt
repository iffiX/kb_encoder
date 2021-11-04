cmake_minimum_required(VERSION 3.12)
project(matcher)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
    message( SEND_ERROR "Compiler version too low, requires g++ > 8.0 to compile" )
endif()
################################################################
# Configuration for depended libraries
################################################################
find_package(HDF5 REQUIRED)
find_package(Python3 COMPONENTS Development)
#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()
add_subdirectory(pybind11)
#set(XTENSOR_USE_XSIMD 1)
################################################################
# Main builds
################################################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/xtl/include)
pybind11_add_module(matcher SHARED init.cpp matcher.cpp concept_net.cpp)
target_include_directories(matcher PUBLIC
        ${Python3_INCLUDE_DIRS}
        ${PYBIND11_INCLUDE_DIR}
        ${HDF5_INCLUDE_DIRS})
target_link_libraries(matcher PRIVATE ${HDF5_LIBRARIES})
install(TARGETS matcher LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

add_library(matcher_test matcher.cpp concept_net.cpp)
target_include_directories(matcher_test PUBLIC
        ${Python3_INCLUDE_DIRS}
        ${PYBIND11_INCLUDE_DIR}
        ${HDF5_INCLUDE_DIRS})
target_link_libraries(matcher_test PUBLIC ${HDF5_LIBRARIES})

add_executable(test main.cpp)
target_link_libraries(test matcher_test)
