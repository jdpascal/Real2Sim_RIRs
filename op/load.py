import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import _import_module_from_library, load


def load_and_cache(name: str, sources):
    """
    Loads a CUDA C++ module and saves the built binaries so it can be loaded directly
    in future runs without recompiling.

    Checkout: https://github.com/pytorch/pytorch/issues/124454#issuecomment-2410055898
    """
    module_path = Path(__file__).parent
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    device_name = torch.cuda.get_device_name()
    device_slug = f"{device_name.replace(' ', '_').lower()}_{cuda_major}_{cuda_minor}"

    build_directory = module_path / "build" / device_slug / name
    build_directory.mkdir(parents=True, exist_ok=True)

    try:
        module = _import_module_from_library(name, build_directory, True)
    except ImportError:
        sources = [os.path.join(module_path, source) for source in sources]
        module = load(
            name,
            sources,
            build_directory=build_directory,
            verbose=True,
            with_cuda=True,
            extra_cflags=["-O3"],
        )
    return module
