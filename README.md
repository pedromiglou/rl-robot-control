# rl-robot-control

## Set up developmente environment

```
docker build -t pedromiglou/gymnasium .
```

```
SHELL=/bin/bash distrobox create --image pedromiglou/gymnasium --name gym-distrobox --additional-flags "--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -m 32G --memory-swap 32G --cpus 4" --home $HOME/gym-distrobox
```
