
# Demo Guide
## Replay the demo
```bash
python src/scripts/replay_demo.py --enable_cameras --task PickCube env.scene.num_envs=4
```
You can also use other tasks. We currently support:
- `PickCube`
- `PlugCharger`
- `CloseBox`
- `StackCube`

## Collect the demo
```bash
python src/scripts/collect_demo.py --enable_cameras --task PickCube env.scene.num_envs=4 --headless
```
The collected demos will be saved under `./data_isaaclab/demo/{task}/robot-franka`.

When collecting demo, the RAM will grow up, during which the collected rendered data are gathered before writing to the disk. After a while, the RAM occupation should become steady.

On RTX 4090, the ideal num_envs is 64.
