# rl-robot-control

This repository contains a simulator inspired by the [Fetch Reach Environment from Gymnasium Robotics](https://robotics.farama.org/envs/fetch/reach/). It also contains some code on how to train a compatible SAC model and how to integrate it in the ROS environment.

## Set up developmente environment with docker and distrobox

```
docker build -t pedromiglou/gymnasium .
```

```
SHELL=/bin/bash distrobox create --image pedromiglou/gymnasium --name gym-distrobox --additional-flags "--gpus all" --home $HOME/gym-distrobox
```
