project('primate2',
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
OS_platform = host_machine.system()
env = environment()

## Configure C++ compiler
cpp = meson.get_compiler('cpp')
# cy = meson.get_compiler('cython')
use_openmp = get_option('use_openmp') and OS_platform != 'windows' and host_machine.cpu() == 'x86_64'
if use_openmp
  add_global_arguments('-DOMP_MULTITHREADED=1', language : 'cpp')
  message('Compiling with OpenMP support')
endif
message('C++ compiler = '+cpp.get_id()+', version: '+cpp.version())
_cpp_args = cpp.get_supported_arguments(
  '-Wno-unused-but-set-variable',
  '-Wno-unused-function',
  '-Wno-conversion',
  '-Wno-misleading-indentation',
  '-Wno-unused-lambda-capture',
  '-Wno-unused-parameter', 
  '-fPIC', 
  '-fvisibility=hidden',
  # '-flto',             # release only: monolithic LTO 
  '-Wno-deprecated-anon-enum-enum-conversion' # to silence Eigen 3.4 warnings
)

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

## Debug build 
_cpp_args += cpp.get_supported_arguments(
  '-O2',
  # '-fsanitize=address', 
  # '-fno-omit-frame-pointer',
  '-g',
  '-Wall'
)

## Release only 
# _cpp_args += compiler.get_supported_arguments(
#   # '-flto=thin',
#   # '-flto', 
#   '-O3',          # full optimizations
#   '-DNDEBUG',     # remove assertions
#   '-Wl,-s',       # strip symbols to reduce binary size
#   # '-march=native' # either this or lto seems to not work on multiple builds
# )

## Require C++20 concepts support
# concept_code = '''
# #include <concepts>
# #include <type_traits>
# template <class T>
# concept integral = std::is_integral_v<T>;
# '''
# concepts_check = cpp.compiles(concept_code, name : 'concepts check')
# if concepts_check
_cpp_args += cpp.get_supported_arguments('-std=c++20')
# else 
#   error('A C++20-compliant compiler is required')
# endif

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

## Add dependencies, if they are available
dependency_map = {}

## Include OpenMP (mandatory ; but exclude on windows because it's too difficult to link) 
if use_openmp
  omp = dependency('openmp', required: false)
  openmp_flags = []
  if omp.found()
    ## Handle msvc, gcc, clang, or clang-cl
    if OS_platform == 'windows' and cpp.get_id() == 'msvc'
      # https://devblogs.microsoft.com/cppblog/improved-openmp-support-for-cpp-in-visual-studio/
      openmp_flags += cpp.get_supported_arguments('/openmp', '/openmp:experimental', '/openmp:llvm')
    elif cpp.get_id() == 'gcc'
      openmp_flags += cpp.get_supported_arguments('-fopenmp', '-fopenmp=libgomp', '-lgomp')
    elif cpp.get_id() == 'clang-cl'
      openmp_flags += cpp.get_supported_arguments('-Xclang', '-fopenmp') 
    elif cpp.get_id() == 'llvm' or cpp.get_id() == 'clang'
      openmp_flags += cpp.get_supported_arguments('-fopenmp', '-fopenmp=libomp', '-lomp')
    endif 
    add_project_link_arguments(openmp_flags, language : 'cpp')
    dependency_map += { 'OpenMP' : omp }
  elif cpp.has_argument('-fopenmp')
    _c = cpp.has_argument('-fopenmp', required: false)
    _h = cpp.has_header('omp.h', required: true)
    omp = declare_dependency(compile_args:'-fopenmp', link_args:'-fopenmp')
    dependency_map += { 'OpenMP' : omp }
  else
    error('OpenMP not detected in the system.')
  endif
endif 


## Configure Pythran include
incdir_pythran = run_command(py,
	['-c', 'import os; os.chdir(".."); import pythran; print(pythran.get_include())'], 
	check: true
).stdout().strip()
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