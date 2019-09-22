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

include_directories(
    /usr/include/
    /usr/include/eigen3
    /usr/local/include/
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


###############################################
# generate separationExtraction executable
set( 
	SEPARATION_SRC
   )
file( GLOB SEPARATION_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/SeparationExtraction/*.cpp)  
add_executable(separationExtraction ${SEPARATION_SRC})
target_link_libraries(separationExtraction ${ALL_LIBS})

###############################################

###############################################
# generate separationExtraction executable
set( 
	CLUSTERING_SRC
   )
file( GLOB CLUSTERING_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/StreamlineClustering/*.cpp)  
add_executable(streamlineClustering ${CLUSTERING_SRC})
target_link_libraries(streamlineClustering ${ALL_LIBS})
###############################################

###############################################
# generate localScalar executable
set( 
	LOCAL_SCALAR_SRC
   )
file( GLOB LOCAL_SCALAR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/LocalScalar/*.cpp)  
add_executable(localScalar ${LOCAL_SCALAR_SRC})

target_link_libraries(localScalar ${ALL_LIBS})
###############################################

###############################################
# generate localScalar executable
set( 
	LOCAL_SCALAR_SRC
   )
file( GLOB LOCAL_SCALAR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/ParallelVector/*.cpp)  
add_executable(pvOperator ${LOCAL_SCALAR_SRC})

target_link_libraries(pvOperator ${ALL_LIBS})
###############################################

###############################################
# generate streamlineGenerator executable
set( 
	LOCAL_SCALAR_SRC
   )
file( GLOB LOCAL_SCALAR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/StreamlineGenerator/*.cpp)  
add_executable(streamlineGenerator ${LOCAL_SCALAR_SRC})

target_link_libraries(streamlineGenerator ${ALL_LIBS})
###############################################