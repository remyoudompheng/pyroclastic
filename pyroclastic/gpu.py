import importlib
import json
import logging
import os
import subprocess


def compile(src, defines, entry="main"):
    with importlib.resources.path(__name__, src) as srcpath:
        defines = [f"-D{k}={v}" for k, v in defines.items()]
        p = subprocess.Popen(
            ["glslc", "--target-env=vulkan1.3", "-I."]
            + defines
            + [f"-fentry-point={entry}", "-fshader-stage=compute", str(srcpath), "-o-"],
            cwd=os.path.dirname(srcpath),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        logging.debug(f"Exec {' '.join(p.args)}")
        out, err = p.communicate(timeout=1)
    if p.returncode:
        raise EnvironmentError(f"shader compilation failed: code {p.returncode}")
    return out


_cached_info = None


def device_info():
    global _cached_info
    if _cached_info:
        return _cached_info
    out = subprocess.check_output("vulkaninfo -j -o /dev/stdout", shell=True)
    _cached_info = json.loads(out)
    return _cached_info


def stamp_period():
    devinfo = device_info()
    props = devinfo["capabilities"]["device"]["properties"]
    devname = props["VkPhysicalDeviceProperties"]["deviceName"]
    return props["VkPhysicalDeviceProperties"]["limits"]["timestampPeriod"]


def max_shmem():
    devinfo = device_info()
    props = devinfo["capabilities"]["device"]["properties"]
    devname = props["VkPhysicalDeviceProperties"]["deviceName"]
    return props["VkPhysicalDeviceProperties"]["limits"]["maxComputeSharedMemorySize"]
