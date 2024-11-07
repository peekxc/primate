## Create a list of the final dependency objects
deps = []
foreach dep_name: dependency_map.keys()
  deps += [dependency_map.get(dep_name)]
endforeach

# xsimd_dep = dependency('xsimd', required: false)

# Define environment for Pythran including custom .pythranrc
# pythran_env = environment()
# pythranrc_path = join_paths(meson.project_source_root(), '.pythranrc')
# pythran_env.set('PYTHRANRC', pythranrc_path)

## Generator to transpile
pythran_gen = generator(
	pythran,
	output : '@BASENAME@.cpp',
	arguments : ['-E', '@INPUT@', '-o', '@OUTPUT@'],
)
## Transpile to C++ 
# run_command(['pythran', '--config', 'compiler.blas=pythran-openblas', '-E', 'tqli.py', '-o', 'tqli.cpp'], check: true)

pythran_modules = [
	'tqli', 
	'fttr', 
]
foreach pythran_ext : pythran_modules
	message('Processing pythran module: ' + pythran_ext)
	py_extension = py.extension_module(
		pythran_ext, 
		pythran_gen.process(pythran_ext + '.py'),
		# include_directories: [openblas_inc],
		# link_args: openblas_link_args,
		cpp_args: [_cpp_args],
		dependencies: [np_dep, pythran_dep],
		install: true, 
		subdir: 'primate'
	)
endforeach

## Manually configured C++ / non-pythran extensions
extensions = [ '_lanczos']
foreach ext: extensions
	inc_pkg = include_directories('include')
	py_extension = py.extension_module(
		'_lanczos', 
		'_lanczos.cpp',
		include_directories: [inc_pkg, inc_numpy, inc_eigen, inc_ffi], # openblas_inc
		# link_args: openblas_link_args,
		cpp_args: [_cpp_args],
		dependencies: [np_dep],
		install: true, 
		subdir: 'primate'
	)
endforeach

## Install python source files
python_sources = [
	'lanczos.py',
	'fttr.py', 
	'tqli.py', 
	'tridiag.py',
	'quadrature.py',
	'stochastic.py',
	'estimators.py',
	'stats.py',
	'operators.py',
	'special.py',
  '__init__.py'
]
message('Source install dir = '+py.get_install_dir())
py.install_sources(
  python_sources,
  subdir: 'primate', 
  pure: false       # Will be installed next to binaries
)

## Install header files
include_sources = [
  'include' / 'lanczos.h',
  'include' / 'linear_operator.h',
	'include' / 'pylinop.h'
]
py.install_sources(
  include_sources,
  subdir: 'primate' / 'include', 
  pure: false       # Will be installed next to binaries
)