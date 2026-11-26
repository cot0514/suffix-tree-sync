# tree_batch 모듈 빌드 (WSL2). 직접 부르지 말고 vllm_path/build.sh 로 빌드한다.
cmake_minimum_required(VERSION 3.18)
project(TreeBatch CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "빌드 종류" FORCE)
endif()

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Threads REQUIRED)

# 설치된 nanobind 위치를 파이썬에서 받아 온다
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# tree_bench 와 같은 소스를 쓴다: Arctic 원본 + trees/ 아래 모든 방식
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
file(GLOB_RECURSE TREE_SOURCES CONFIGURE_DEPENDS "${REPO_ROOT}/trees/*.cc")

nanobind_add_module(tree_batch NOMINSIZE
  batch_module.cc
  "${REPO_ROOT}/third_party/arctic/suffix_tree.cc"
  ${TREE_SOURCES})
target_link_libraries(tree_batch PRIVATE Threads::Threads)
