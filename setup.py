"""Setup file for scranpy. Use setup.cfg to configure your project.

This file was generated with PyScaffold 4.5.
PyScaffold helps you to put up the scaffold of your new Python project.
Learn more under: https://pyscaffold.org/
"""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig
import pathlib
import os
import shutil
import sys


def build_igraph(builder):
    if "SCRANPY_INSTALLED_PATH" in os.environ:
        install_dir = os.environ["SCRANPY_INSTALLED_PATH"]
    else:
        install_dir = os.path.join(os.getcwd(), "installed")
    if os.path.exists(install_dir):
        return install_dir

    if builder is None:
        import subprocess
        class Tmp:
            def __init__(self):
                pass
            def spawn(self, cmd):
                subprocess.run(cmd, check=True)
        builder = Tmp()
    else:
        raise ValueError("ARRGHH")

    version = "1.0.0"
    if not os.path.exists("extern"):
        os.mkdir("extern")

    src_dir = os.path.join("extern", "igraph-" + version)
    if not os.path.exists(src_dir):
        tarball = os.path.join("extern", "igraph.tar.gz")
        if not os.path.exists(tarball):
            import urllib.request
            target_url = " https://github.com/igraph/igraph/releases/download/" + version + "/igraph-" + version + ".tar.gz"
            urllib.request.urlretrieve(target_url, tarball)
        import tarfile
        with tarfile.open(tarball, "r") as tf:
            tf.extractall("extern")

    build_dir = os.path.join("extern", "build-" + version)
    os.mkdir("installed")

    cmd = [
        "cmake",
        "-S", src_dir,
        "-B", build_dir,
        "-DCMAKE_POSITION_INDEPENDENT_CODE=true",
        "-DIGRAPH_WARNINGS_AS_ERRORS=OFF",
        "-DCMAKE_INSTALL_PREFIX=" + install_dir,
        "-DIGRAPH_USE_INTERNAL_GMP=ON",
        "-DIGRAPH_USE_INTERNAL_BLAS=ON",
        "-DIGRAPH_USE_INTERNAL_LAPACK=ON",
        "-DIGRAPH_USE_INTERNAL_ARPACK=ON",
        "-DIGRAPH_USE_INTERNAL_GLPK=ON",
        "-DIGRAPH_USE_INTERNAL_GMP=ON",
        "-DIGRAPH_USE_INTERNAL_PLFIT=ON",
        "-DIGRAPH_ENABLE_LTO=ON",
        "-DIGRAPH_OPENMP_SUPPORT=OFF",
    ]
    if os.name != "nt":
        cmd.append("-DCMAKE_BUILD_TYPE=Release")
    if "MORE_CMAKE_OPTIONS" in os.environ:
        cmd += os.environ["MORE_CMAKE_OPTIONS"].split()
    builder.spawn(cmd)

    cmd = ['cmake', '--build', build_dir]
    if os.name == "nt":
        cmd += ["--config", "Release"]
    builder.spawn(cmd)
    cmd = ['cmake', '--install', build_dir]
    builder.spawn(cmd)
    return install_dir


## Adapted from https://stackoverflow.com/questions/42585210/extending-setuptools-extension-to-use-cmake-in-setup-py.
class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        install_dir = build_igraph(self)

        build_temp = pathlib.Path(self.build_temp)
        build_lib = pathlib.Path(self.build_lib)
        outpath = os.path.join(build_lib.absolute(), ext.name) 

        if not os.path.exists(build_temp):
            import assorthead
            import mattress 
            import knncolle 
            import pybind11

            cmd = [ 
                "cmake", 
                "-S", "lib",
                "-B", build_temp,
                "-Dpybind11_DIR=" + os.path.join(os.path.dirname(pybind11.__file__), "share", "cmake", "pybind11"),
                "-DPYTHON_EXECUTABLE=" + sys.executable,
                "-DASSORTHEAD_INCLUDE_DIR=" + assorthead.includes(),
                "-DMATTRESS_INCLUDE_DIR=" + mattress.includes(),
                "-DKNNCOLLE_INCLUDE_DIR=" + knncolle.includes(),
                "-DCMAKE_PREFIX_PATH=" + install_dir
            ]
            if os.name != "nt":
                cmd.append("-DCMAKE_BUILD_TYPE=Release")
                cmd.append("-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + outpath)

            if "MORE_CMAKE_OPTIONS" in os.environ:
                cmd += os.environ["MORE_CMAKE_OPTIONS"].split()
            self.spawn(cmd)

        if not self.dry_run:
            cmd = ['cmake', '--build', build_temp]
            if os.name == "nt":
                cmd += ["--config", "Release"]
            self.spawn(cmd)
            if os.name == "nt": 
                # Gave up trying to get MSVC to respect the output directory.
                # Delvewheel also needs it to have a 'pyd' suffix... whatever.
                shutil.copyfile(os.path.join(build_temp, "Release", "_core.dll"), os.path.join(outpath, "_core.pyd"))


if __name__ == "__main__":
    import os
    try:
        setup(
            use_scm_version={"version_scheme": "no-guess-dev"},
            ext_modules=[CMakeExtension("scranpy")],
            cmdclass={
                'build_ext': build_ext
            }
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
