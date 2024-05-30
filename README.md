# rl-robot-control

## Set up developmente environment

```
docker build -t pedromiglou/gymnasium .
```

```
SHELL=/bin/bash distrobox create --image pedromiglou/gymnasium --name gym-distrobox --additional-flags "--gpus all" --home $HOME/gym-distrobox
```
