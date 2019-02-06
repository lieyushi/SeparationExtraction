# CMake entry point
cmake_minimum_required (VERSION 2.8)

set( component_SOURCES  
	 *.h
	 *.cpp
   ) # Add the source-files for the component here

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

if(${VTK_VERSION} VERSION_GREATER "6" AND VTK_QT_VERSION VERSION_GREATER "4")
	# Instruct CMake to run moc automatically when needed.
	set(CMAKE_AUTOMOC ON)
	find_package(Qt5Widgets REQUIRED QUIET)
else()
	find_package(Qt4 REQUIRED)
	include(${QT_USE_FILE})
endif()

include_directories(
    /usr/include/
    /usr/include/eigen3
    /usr/local/include/
    /usr/local/include/vtk-8.1/
	./src
    .
)

file( GLOB component_SOURCES *.cpp )

add_library( common ${component_SOURCES} )