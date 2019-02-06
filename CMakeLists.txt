# CMake entry point
cmake_minimum_required (VERSION 2.8)
project (MyClustering)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/Common)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# add "-O2" for library eigen optimization for matrix computation.
# "-O2" is C++ optimization flags (bigO 2)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")

# find_package(VTK REQUIRED)
# include(${VTK_USE_FILE})

include_directories(
    /usr/include/
    /usr/include/eigen3
    /usr/local/include/
    /usr/local/include/vtk-8.1/
    src/Common/
	./src
    .
)

set(ALL_LIBS
	common
)

add_definitions(
#	-DTW_STATIC
#	-DTW_NO_LIB_PRAGMA
#	-DTW_NO_DIRECT3D
#	-DGLEW_STATIC
#	-D_CRT_SECURE_NO_WARNINGS
)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

###############################################
# generate separationExtraction executable
set( 
	SEPARATION_SRC
   )
file( GLOB SEPARATION_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/SeparationExtraction/*.cpp)  
add_executable(separationExtraction ${SEPARATION_SRC})
if(VTK_LIBRARIES)
	target_link_libraries(separationExtraction ${VTK_LIBRARIES} ${ALL_LIBS})
else()
	target_link_libraries(separationExtraction vtkHybrid vtkWidgets ${ALL_LIBS})
endif()
###############################################