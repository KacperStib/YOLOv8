FROM ros:humble-perception

ENV WS_DIR="/ros2_ws"
WORKDIR ${WS_DIR}

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

RUN useradd -m -s /bin/bash -u 1000 rosdev && \
    echo "rosdev ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN apt-get update \
 && apt-get install -y \
    build-essential \
    cmake \
    git-all \
    software-properties-common \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*


RUN apt-get update \
 && apt-get install -y \
    ros-${ROS_DISTRO}-librealsense2* \
    ros-${ROS_DISTRO}-realsense2-* \
    ros-${ROS_DISTRO}-rviz2 \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install pyrealsense2 numpy opencv-python ultralytics

ARG DEBIAN_FRONTEND=dialog