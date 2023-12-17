## Create a list of the final dependency objects
deps = []
foreach dep_name: dependency_map.keys()
  deps += [dependency_map.get(dep_name)]
endforeach

## Extension modules
extensions = [ '_random_gen', '_operators', '_orthogonalize', '_trace', '_lanczos']
foreach ext: extensions
  py.extension_module(
    ext,
    ext + '.cpp',
    include_directories: [inc_pkg, inc_numpy, inc_eigen, inc_pybind11], # inc_pybind11, inc_nanobind
    dependencies: deps,
    install: true, 
    cpp_args: _cpp_args
  )
endforeach

python_sources = [
  'random.py',
  'ortho.py', 
  'diagonalize.py',
  'plotting.py',
  'trace.py',
  'operator.py', 
  'stats.py',
  'special.py',
  '__init__.py'
]

## Print install directory
message('Source install dir = '+py.get_install_dir())
py.install_sources(
  python_sources,
  subdir: 'primate', 
  pure: false       # Will be installed next to binaries
)