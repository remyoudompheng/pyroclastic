import importlib
import json
import logging
import os
import subprocess


def compile(src, defines, entry="main"):
    with importlib.resources.path("pyroclastic", src) as srcpath:
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


class GPUInfo:
    def __init__(self):
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        self.loaded = True
        try:
            self.vulkaninfo()
            logging.info("Loaded information from vulkaninfo")
        except Exception:
            pass

    def vulkaninfo(self):
        out = subprocess.check_output("vulkaninfo -j -o /dev/stdout", shell=True)
        info = json.loads(out)
        if "VkPhysicalDeviceProperties" in info:
            phyprops = info["VkPhysicalDeviceProperties"]
        else:
            props = info["capabilities"]["device"]["properties"]
            phyprops = props["VkPhysicalDeviceProperties"]
        self.devname = phyprops["deviceName"]
        self.devtype = phyprops["deviceType"]
        self.stamp_period = phyprops["limits"]["timestampPeriod"]
        self.max_shmem = phyprops["limits"]["maxComputeSharedMemorySize"]


_gpuinfo = GPUInfo()


def device_name() -> str:
    _gpuinfo.load()
    return _gpuinfo.devname


def is_discrete_gpu() -> bool:
    _gpuinfo.load()
    return _gpuinfo.devtype == "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"


def stamp_period() -> int:
    _gpuinfo.load()
    return _gpuinfo.stamp_period


def max_shmem() -> int:
    _gpuinfo.load()
    return _gpuinfo.max_shmem


def has_fast_add64() -> bool:
    """
    Whether the devices supports fast int64 add/sub operations
    """
    # FIXME: test other vendors
    # Currently only NVIDIA GPU run int64 kernels faster
    _gpuinfo.load()
    return "nvidia" in _gpuinfo.devname.lower()
