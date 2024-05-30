# Use distrobox image with ROS Noetic and CUDA 11.8 as base
FROM pedromiglou/ros-cuda-distrobox:noetic-11.8

# Install apt dependencies
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev libgl1-mesa-glx \
        libglew-dev libosmesa6-dev patchelf python3-pip \
        libglfw3 python3-tk

# pytorch by default tries to install a networkx version that
# needs python 3.9
RUN pip install networkx
# Install pytorch with cuda 11.8 and other python dependencies
RUN pip install torch --index-url \
        https://download.pytorch.org/whl/cu118
RUN pip install gymnasium gymnasium-robotics \
        stable-baselines3 pygame moviepy \
        matplotlib pandas numpy==1.24.4
