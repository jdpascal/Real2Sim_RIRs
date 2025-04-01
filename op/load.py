import os

from torch.utils.cpp_extension import load, _import_module_from_library

def load_and_cache(name, sources):
    """
    Loads a CUDA C++ module and saves the built binaries so it can be loaded directly
    in future runs without recompiling.
    
    Checkout: https://github.com/pytorch/pytorch/issues/124454#issuecomment-2410055898
    """
    module_path = os.path.dirname(__file__)
    build_directory = os.path.join(module_path, "build", name)
    os.makedirs(build_directory, exist_ok=True)

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
