cmake_minimum_required(VERSION 3.5)
project(z_arm_2442c0_model)

# 默认使用 C++14（如果你有 C++ 代码）
set(CMAKE_CXX_STANDARD 14)

# 查找依赖包（根据你实际使用的 ROS 2 包来添加）
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(moveit REQUIRED)


# 如果你有源代码
# add_executable(my_node src/my_node.cpp)
# add_executable(planningDemo launch/planningDemo.py)
# ament_target_dependencies(my_node rclcpp std_msgs)

# 安装 launch、urdf、meshes 等资源文件
install(DIRECTORY src/launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY src/urdf DESTINATION share/${PROJECT_NAME})
install(DIRECTORY src/meshes DESTINATION share/${PROJECT_NAME})
# install(DIRECTORY moveit DESTINATION share/${PROJECT_NAME})
install(DIRECTORY src/config DESTINATION share/${PROJECT_NAME})

# 如果你有节点可执行文件
# install(TARGETS my_node
#   DESTINATION lib/${PROJECT_NAME}
# )

ament_package()
