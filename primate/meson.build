## Extension modules
# inc_blas, inc_lapack, 
extensions = [ '_random_generator', '_trace', '_diagonalize', '_orthogonalize', '_sparse_eigen' ]
# extensions = [ '_random_generator' ]
foreach ext: extensions
  py.extension_module(
    ext,
    ext + '.cpp',
    include_directories: [inc_pkg, incdir_numpy, inc_pybind11, inc_eigen],
    dependencies: [lapack, blas, omp],
    install: true, 
    cpp_args: _cpp_args
  )
endforeach

python_sources = [
  '__init__.py',
  'trace.py', 
  'ortho.py'
]

## Print install directory
message('Source install dir = '+py.get_install_dir())
py.install_sources(
  python_sources,
  subdir: 'primate', 
  pure: false                              # Will be installed next to binaries
)