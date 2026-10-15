"""
vusvna, Vulkan Self-Validated Numerical Analysis: Slang module for rigorous affine arithmetic in Vulkan
shaders. The Python package only ships the Slang sources, add get_include() to the include paths of the
Slang compiler (slangpy: SlangCompilerOptions({"include_paths": [vusvna.get_include()]})) and write
`import vusvna;`.
"""

import os

__version__ = "1.0.1"

_package_dir = os.path.dirname(os.path.abspath(__file__))


def get_include():
    """Directory containing vusvna.slang, to add to the include paths of the Slang compiler."""
    slang_dir = os.path.join(_package_dir, "slang")
    if os.path.isfile(os.path.join(slang_dir, "vusvna.slang")):
        return slang_dir
    # source checkout or editable install: the sources are in the vusvna/ directory of the repository
    return os.path.join(os.path.dirname(os.path.dirname(_package_dir)), "vusvna")
