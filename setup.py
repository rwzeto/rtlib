from os import *

target_dir = "target\\debug\\"

files = listdir(target_dir)

if "rtlib.dll" in files and "rtlib.pyd" in files:
    remove(target_dir+"rtlib.pyd")
    rename(target_dir+"rtlib.dll", target_dir+"rtlib.pyd")
elif "rtlib.dll" in files and "rtlib.pyd" not in files:
    rename(target_dir+"rtlib.dll", target_dir+"rtlib.pyd")
elif "rtlib.dll" not in files and "rtlib.pyd" in files:
    pass
else:
    "no library exists, run cargo build."