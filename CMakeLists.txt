##############################################################################
# CMAKE CONFIGURATION
##############################################################################
cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

project(obstructionfre VERSION 1 LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Release)

message("\n" "=========================================")
message("Project: ${PROJECT_NAME} ")
message("=========================================")

# set corresponding package directories
set(OpenCV_DIR /opt/opencv-4.1.2/build)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 11)

##############################################################################
# LIBRARIES CONFIG FILE
##############################################################################
message("***********************")
message("OpenCV Library")
message("***********************")

find_package(OpenCV 4 PATHS ${OpenCV_DIR} QUIET)
if(OpenCV_FOUND)
   	message(STATUS "OpenCV status:")
   	message(STATUS "    version: ${OpenCV_VERSION}")
else()
    message(WARNING "   OpenCV 4 not found, attempting 3...")
    find_package(OpenCV 3.4.1 REQUIRED QUIET)
    if(OpenCV_FOUND)
        message(STATUS "OpenCV status:")
   		message(STATUS "    version: ${OpenCV_VERSION}")
    else()
        message(FATAL_ERROR "    ERROR: OpenCV ${OpenCV_VERSION} not found")
    endif()
endif()

##############################################################################
# LIBRARIES HEADERS
##############################################################################
include_directories(
  ${OpenCV_INCLUDE_DIRS}  
)

include(CheckFunctionExists)

##############################################################################
# LIBRARIES PATH
##############################################################################
link_directories(
	${OpenCV_LIBRARIES}	                
)

##############################################################################
# SOURCE CODE
##############################################################################
set(MAIN_SOURCE "main.cpp"
)

##############################################################################
# EXECUTABLES
##############################################################################
add_executable(${PROJECT_NAME}             
               ${MAIN_SOURCE}
)
               
##############################################################################
# TARGET LIBRARIES
##############################################################################
target_link_libraries(${PROJECT_NAME}
                      ${OpenCV_LIBS}                                                              
)                

message("=========================================")
message("Project: ${PROJECT_NAME} COMPILED WITH CMAKE " ${CMAKE_VERSION})
message("=========================================")
