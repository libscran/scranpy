import os

def build_igraph(builder):
    if "SCRANPY_INSTALLED_PATH" in os.environ:
        install_dir = os.environ["SCRANPY_INSTALLED"]
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


if __name__ == "__main__":
    build_igraph(None)
