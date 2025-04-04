
# Trouble Shooting
## GPU memory problems with IsaacLab
If you interrupt the process, the GPU memory may not be released because **the forked process is not killed**. You need to manually kill it by running:
```bash
pkill -9 pt_main_thread
```


## No Space XXX
- If you meet `Failed to create change watch for ... No space left on device`, it doesn't really matter. You can simply ignore it, or change your system setting by running `echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf` or just simply restart the system.

## libGLU problem
- If you are running on a server in headless mode, and meet `Failed to open /.../iray/libneuray.so: libGLU.so.1: cannot open shared object file: No such file or directory`, try:
    ```bash
    sudo apt-get update
    sudo apt-get install libglu1-mesa
    ```

## USD Missing
- If you load a USD to the scene but is invisible,
  It might be that the USD path is not correctly set. The safest way is to use absolute path, and to make the code clean, please use the `os.path.abspath` to convert the relative path to absolute path. For example:
```python
usd_path=os.path.abspath(f"data_isaaclab/robots/FrankaEmika/panda_instanceable.usd"),
```
