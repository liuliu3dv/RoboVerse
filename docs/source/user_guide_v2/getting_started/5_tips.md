# Tips
## Run with limited CPU cores
You can limit the CPU cores used by Isaac Sim by setting `taskset`. For example, if you want to use 0,1,2,3 cores, you can run:
```bash
taskset -c 0,1,2,3 python XXXX # scripts
```

##  Hydra system
You can overwrite the `*EnvCfg` from command line. For example, if you want to set `PickCubeEnvCfg.decimation` to `3`, you can run:
```bash
python src/scripts/demo.py --enable_cameras --task PickCube env.scene.num_envs=4 env.decimation=3
```
For more details, please refer to [the official guide](https://isaac-sim.github.io/IsaacLab/main/source/features/hydra.html).

## Sync data with Google Drive
We recommend using rclone to sync the data with google drive.

### Setup rclone
Create a rclone config under the [official guide](https://rclone.org/drive/), during which you need to create your own client id and token by following [this](https://rclone.org/drive/#making-your-own-client-id).

The final rclone config should look like this:
```
[roboverse]
type = drive
client_id = <your_client_id>
client_secret = <your_client_secret>
scope = drive
token = <your_token>
team_drive =
root_folder_id = 1ORMP3__KIlXettN8eUCF3YQNybZQxzkw

[roboverse_isaaclab]
type = drive
client_id = <your_client_id>
client_secret = <your_client_secret>
scope = drive
token = <your_token>
team_drive =
root_folder_id = 1nF-5SU4nC6S_vgC_NL7E7Rt63Zl9sYqW
```

### Download data
```bash
rclone copy -P roboverse: data --exclude='demo/**'
rclone copy -P roboverse_isaaclab: data_isaaclab
```
You may want to speed up by setting `--multi-thread-streams=16` (default is 4).

### Upload data (Developer only)
For example, if you are a developer who is migrating the rlbench task, you can upload the assets and source demos to google drive by running:
```bash
rclone copy -P data/assets/rlbench roboverse_isaaclab:assets/rlbench --exclude='.thumbs/**'
rclone copy -P data/source_data/rlbench roboverse:data/source_data --include='trajectory-unified.pkl'
```
You ***should*** replace `-P` with `-n` (means dry run) to check before uploading.
