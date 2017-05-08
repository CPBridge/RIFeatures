# You may use this CMake file to include the RIFeatures library in other
# projects using CMake

# Dependency on OpenMP (optional)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Dependency on Boost libraries (header only parts)
find_package( Boost REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} )

# Dependency on OpenCV
find_package( OpenCV REQUIRED )
add_library(RIFeatures src/RIFeatExtractor.cpp src/struve.cpp)
target_link_libraries( RIFeatures ${OpenCV_LIBS} )
