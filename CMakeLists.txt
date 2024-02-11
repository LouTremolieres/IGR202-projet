cmake_minimum_required(VERSION 3.5)

SET(CMAKE_EXPORT_COMPILE_COMMANDS 1)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

project(projet)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_executable(
  projet
  src/main.cpp
  src/Error.cpp
  src/Mesh.cpp
  src/ShaderProgram.cpp)


add_subdirectory(dep/glad)
target_link_libraries(${PROJECT_NAME} PRIVATE glad)

add_subdirectory(dep/glfw)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

add_subdirectory(dep/glm)
target_link_libraries(${PROJECT_NAME} PRIVATE glm)

target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS})

add_custom_command(TARGET projet
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:projet> ${CMAKE_CURRENT_SOURCE_DIR})
