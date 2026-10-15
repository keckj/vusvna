# The Python package only ships the Slang module: the sources are copied from vusvna/ to vusvna/slang/ in the
# wheel, vusvna.get_include() returns the directory to add to the include paths of the Slang compiler. All
# metadata lives in pyproject.toml.
import glob
import os

from setuptools import setup
from setuptools.command.build_py import build_py

root = os.path.dirname(os.path.abspath(__file__))


class build_py_with_slang(build_py):
    def run(self):
        super().run()
        sources = sorted(glob.glob(os.path.join(root, "vusvna", "*.slang")))
        if not sources:
            raise RuntimeError("vusvna Slang sources not found")
        target = os.path.join(self.build_lib, "vusvna", "slang")
        self.mkpath(target)
        for source in sources:
            self.copy_file(source, os.path.join(target, os.path.basename(source)))


setup(cmdclass={"build_py": build_py_with_slang})
