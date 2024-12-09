project('primate',
	'cpp',
	# 'cpp', 'cython', 
	# version: run_command(['tools/gitversion.py'], check: true).stdout().strip(),
  license: 'Apache',
  meson_version: '>= 1.5.0',
  default_options: [
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
    'cpp_std=c++20', 
		'warning_level=2',
    'pkgconfig.relocatable=true'
  ],
)
project_root = meson.project_source_root()
os_platform = host_machine.system()
dependency_map = {}

## Configure C++ compiler
cpp = meson.get_compiler('cpp')
message('C++ compiler = '+cpp.get_id()+', version: '+cpp.version())

_cpp_args = cpp.get_supported_arguments(
  '-Wno-unused-but-set-variable',
  '-Wno-unused-function',
  '-Wno-conversion',
  '-Wno-misleading-indentation',
  '-Wno-unused-lambda-capture',
  '-Wno-unused-parameter', 
	'-Wno-cast-function-type',
  '-fPIC', 
  '-fvisibility=hidden', # only expose exported definitions; reduces binary size
  '-Wno-deprecated-anon-enum-enum-conversion', # to silence Eigen 3.4 warnings
	'-Wl,-s',       # strip symbols to reduce binary size
)
_link_args = []

## Add C++20 
_cpp_args += cpp.get_supported_arguments('-std=c++20')

## Choose release vs debug mode
if get_option('release')
	_cpp_args += cpp.get_supported_arguments(
		'-O3',          # full optimizations
		'-DNDEBUG',     # remove assertions
		# '-march=native' # either this or lto seems to not work on multiple builds
		'-fno-unroll-loops',	# no unrolled loops
	)
	_link_args += cpp.get_supported_link_arguments(
		'-Wl,-s',       # strip symbols to reduce binary size
	)
	## Add link-time optimization
	## NOTE: this reduces the size of the binaries significantly!
	## see: https://gcc.gnu.org/onlinedocs/gccint/LTO-Overview.html and https://stackoverflow.com/questions/23736507/is-there-a-reason-why-not-to-use-link-time-optimization-lto
	if cpp.get_id() == 'clang'
		_cpp_args += cpp.get_supported_arguments('-flto=thin')
	else
		_cpp_args += cpp.get_supported_arguments('-flto')
	endif 
else
	_cpp_args += cpp.get_supported_arguments(
  	'-O2',
  	# '-fsanitize=address', 
  	# '-fno-omit-frame-pointer',
  	'-g',
  	'-Wall'
	)
endif


## Configure OpenMP, if option configured
subdir('openmp')

## Import python interpreter
py_mod = import('python')
py = py_mod.find_installation(pure: true)
py_dep = py.dependency()
message('Python path =' + py.full_path())
message('Numpy version =' + run_command(py, ['-c', 'import numpy; print(numpy.__version__)'], check: true).stdout().strip())
if py.language_version().version_compare('< 3.8')
  error('Invalid Python version, only >= 3.8 is supported.')
endif

## Configure Numpy include
incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()
inc_np = include_directories(incdir_numpy)
np_dep = declare_dependency(include_directories: inc_np)

## Include directories
incdir_numpy = run_command(py, ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'], check : true).stdout().strip()

## Header includes
inc_numpy = include_directories(incdir_numpy)
inc_eigen = include_directories('extern' / 'eigen')

## Configure FFI
if get_option('use_nanobind')
	ffi_dep = dependency('nanobind', static: true)
	incdir_nanobind = run_command(py, ['-c', 'import os; os.chdir(".."); import nanobind; print(nanobind.include_dir())'], check : true).stdout().strip()
	inc_ffi = include_directories(incdir_nanobind)
else
	incdir_pybind11 = run_command(py, ['-c', 'import os; os.chdir(".."); import pybind11; print(pybind11.get_include())'], check : true).stdout().strip()
	inc_ffi = include_directories(incdir_pybind11)
endif 

## Configure Pythran include
incdir_pythran = run_command(py, ['-c', 'import os; os.chdir(".."); import pythran; print(pythran.get_include())'], check: true).stdout().strip()
pythran_dep = declare_dependency(
  include_directories: incdir_pythran,
)

## Pythran 
pythran = find_program('pythran', native: true, version: '>=0.14.0')
_cpp_args += [
	'-DENABLE_PYTHON_MODULE',
	'-D__PYTHRAN__=3',
	# '-DPYTHRAN_BLAS_NONE'
]

## Pythran specific compiler flags
_cpp_args += cpp.get_supported_arguments(
  '-Wno-unused-but-set-variable',
  '-Wno-unused-function',
  '-Wno-conversion',
  '-Wno-misleading-indentation',
  '-Wno-unused-lambda-capture',
  '-Wno-unused-parameter', 
	'-Wno-deprecated-declarations',
	'-Wno-unused-local-typedef'
)

# openblas_inc = include_directories('/usr/local/opt/openblas/include')
# openblas_lib = '/usr/local/opt/openblas/lib'
# openblas_link_args = ['-L' + openblas_lib, '-lopenblas']

## Compile the package directory
subdir('src' / 'primate')