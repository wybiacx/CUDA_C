# 设置可执行文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/chapter3)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/chapter3)

# 查找当前目录下的所有 CUDA 源文件
file(GLOB CUDA_SOURCES *.cu)

# 为每个 CUDA 源文件创建可执行文件
foreach(CUDA_SOURCE ${CUDA_SOURCES})
    get_filename_component(EXECUTABLE_NAME ${CUDA_SOURCE} NAME_WE)
    # 为目标名称添加前缀，以避免与其他章节冲突
    add_executable(chapter3_${EXECUTABLE_NAME} ${CUDA_SOURCE})
    # 将可执行文件的名称设置为原始名称
    set_target_properties(chapter3_${EXECUTABLE_NAME} PROPERTIES
        RUNTIME_OUTPUT_NAME ${EXECUTABLE_NAME}
    )
endforeach()

# 添加公共 include 目录
include_directories(${CMAKE_SOURCE_DIR}/common)
