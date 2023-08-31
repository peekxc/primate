## Extension modules
# inc_blas, inc_lapack, 
extensions = [ '_random_generator', '_trace', '_diagonalize', '_orthogonalize', '_operators' ]
foreach ext: extensions
  py.extension_module(
    ext,
    ext + '.cpp',
    include_directories: [inc_pkg, inc_imate, incdir_numpy, inc_pybind11],
    dependencies: [lapack, blas, omp],
    install: true, 
    cpp_args: _cpp_args
  )
endforeach

python_sources = [
  '__init__.py',
  'trace.py', 
  'ortho.py', 
]

## Print install directory
message('Source install dir = '+py.get_install_dir())
py.install_sources(
  python_sources,
  subdir: 'pyimate', 
  pure: false                              # Will be installed next to binaries
)

# subdir('_random_generator')