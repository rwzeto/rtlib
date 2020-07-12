from os import *

target_dir = "./target/debug/"

files = listdir(target_dir)

if "librtlib.so" in files and "rtlib.so" in files:
    remove(target_dir+"rtlib.so")
    rename(target_dir+"librtlib.so", target_dir+"rtlib.so")
elif "librtlib.so" in files and "rtlib.so" not in files:
    rename(target_dir+"librtlib.so", target_dir+"rtlib.so")
elif "librtlib.so" not in files and "rtlib.so" in files:
    pass
else:
    "no library exists, run cargo build."