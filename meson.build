project(
  'primate', 'cpp',
  version: run_command(['src/primate/__version__.py'], check: true).stdout().strip(),
  license: 'Apache-2.0',
  meson_version: '>= 1.1.0',
  default_options : [
    'buildtype=debugoptimized', 
    'b_ndebug=if-release',
    'cpp_std=c++20', 
    'warning_level=2',
    # 'blas=openblas',
    # 'lapack=openblas', 
    'pkgconfig.relocatable=true'
  ]
)
OS_platform = host_machine.system()
env = environment()
use_openmp = get_option('use_openmp') and OS_platform != 'windows' and host_machine.cpu() == 'x86_64'

if use_openmp
  add_global_arguments('-DOMP_MULTITHREADED=1', language : 'cpp')
  message('Compiling with OpenMP support')
endif

## Import python kernel 
py_mod = import('python')
py = py_mod.find_installation(pure: true)
py_dep = py.dependency()
message('Python path =' + py.full_path())
message('Numpy version =' + run_command(py, ['-c', 'import numpy; print(numpy.__version__)'], check: true).stdout().strip())

## Check the python version
if py.language_version().version_compare('< 3.8')
  error('Invalid Python version, only >= 3.8 is supported.')
endif

## Print the C++ compiler + tack on default arguments
compiler = meson.get_compiler('cpp')
message('Compiler = '+compiler.get_id()+', version: '+compiler.version())
_cpp_args = compiler.get_supported_arguments(
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

## Debug only 
# _cpp_args += compiler.get_supported_arguments(
#   '-O1',
#   # '-fsanitize=address', 
#   # '-fno-omit-frame-pointer',
#   '-DNDEBUG',
#   '-Wall'
# )

## Release only 
_cpp_args += compiler.get_supported_arguments(
  # '-flto=thin',
  # '-flto', 
  '-O3', 
  '-DNDEBUG',
  '-Wl,-s',       # strip
  # '-march=native' # either this or lto seems to not work on multiple builds
)

## Require C++20 for concepts
concept_code = '''
#include <concepts>
#include <type_traits>
template <class T>
concept integral = std::is_integral_v<T>;
'''
concepts_check = compiler.compiles(concept_code, name : 'concepts check')
if concepts_check
  _cpp_args += compiler.get_supported_arguments('-std=c++20')
else 
  error('A C++20-compliant compiler is required')
endif

## Include directories
incdir_numpy = run_command(py, ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'], check : true).stdout().strip()
# incdir_eigen = run_command(py, ['-c', 'import os; os.chdir(".."); import eigenpip; print(eigenpip.get_include())'], check : true).stdout().strip()
# incdir_imate = run_command(py, ['-c', 'import os; os.chdir(".."); import imate; print(imate.__path__[0])'], check : true).stdout().strip()
# incdir_nanobind = run_command(py, ['-c', 'import os; os.chdir(".."); import nanobind; print(nanobind.include_dir())'], check : true).stdout().strip()
incdir_pybind11 = run_command(py, ['-c', 'import os; os.chdir(".."); import pybind11; print(pybind11.get_include())'], check : true).stdout().strip()

## Header includes
inc_numpy = include_directories(incdir_numpy)
inc_pybind11 = include_directories(incdir_pybind11)
# inc_nanobind = include_directories(incdir_nanobind) # nanobind.include_dir()
inc_eigen = include_directories('extern' / 'eigen')
# inc_eigen = include_directories(incdir_eigen)
# inc_imate = include_directories(incdir_imate)
inc_pkg = include_directories('include')

## Add dependencies, if they are available
dependency_map = {}
# subdir('blas') # Configure BLAS / LAPACK 

## Include OpenMP (mandatory ; but exclude on windows because it's too difficult to link) 
if use_openmp
  omp = dependency('openmp', required: false)
  openmp_flags = []
  if omp.found()
    ## Handle msvc, gcc, clang, or clang-cl
    if OS_platform == 'windows' and compiler.get_id() == 'msvc'
      # https://devblogs.microsoft.com/cppblog/improved-openmp-support-for-cpp-in-visual-studio/
      openmp_flags += compiler.get_supported_arguments('/openmp', '/openmp:experimental', '/openmp:llvm')
    elif compiler.get_id() == 'gcc'
      openmp_flags += compiler.get_supported_arguments('-fopenmp', '-fopenmp=libgomp', '-lgomp')
    elif compiler.get_id() == 'clang-cl'
      openmp_flags += compiler.get_supported_arguments('-Xclang', '-fopenmp') 
    elif compiler.get_id() == 'llvm' or compiler.get_id() == 'clang'
      openmp_flags += compiler.get_supported_arguments('-fopenmp', '-fopenmp=libomp', '-lomp')
    endif 
    add_project_link_arguments(openmp_flags, language : 'cpp')
    dependency_map += { 'OpenMP' : omp }
  elif compiler.has_argument('-fopenmp')
    _c = compiler.has_argument('-fopenmp', required: false)
    _h = compiler.has_header('omp.h', required: true)
    omp = declare_dependency(compile_args:'-fopenmp', link_args:'-fopenmp')
    dependency_map += { 'OpenMP' : omp }
  else
    error('OpenMP not detected in the system.')
  endif
endif 

## Set USE_CBLAS to 0 to use C++ implemented sparse matrix vector operations
## Otherwise BLAS must be linked and used
# env = environment()
# env.set('USE_CBLAS', '1') ## Turn off CBLAS

## Use SciPy-configured BLAS
# scipy_blas_found = run_command(py, ['-c', 'import scipy; scipy.__config__.CONFIG["Build Dependencies"]["blas"]["found"]'], check : false).stdout().strip()
# scipy_blas2_found = run_command(py, ['-c', 'import scipy; "blas_opt_info" in dir(scipy.__config__)'], check : false).stdout().strip()
# message(scipy_blas_found)
# message(scipy_blas2_found)

# if scipy_blas_found == 'true'
#   blas_incdir = run_command(py, ['-c', 'import scipy; scipy.__config__.CONFIG["Build Dependencies"]["blas"]["include directory"]))'], check : true).stdout().strip()
#   blas_libdirs = run_command(py, ['-c', 'import scipy; scipy.__config__.CONFIG["Build Dependencies"]["blas"]["lib directory"]))'], check : true).stdout().strip()
#   blas_libs = ['blas']
#   message('SCIPY BLAS INFORMATION')
#   message(blas_incdir)
#   message(blas_libdirs)
#   message(blas_libs)
# elif scipy_blas2_found == 'true'
#   blas_incdir = run_command(py, ['-c', 'import scipy as sp; print(" ".join(sp.__config__.blas_opt_info["include_dirs"]))'], check : true).stdout().strip()
#   blas_libdirs = run_command(py, ['-c', 'import scipy as sp; print(" ".join(sp.__config__.blas_opt_info["library_dirs"]))'], check : true).stdout().strip()
#   blas_libs = run_command(py, ['-c', 'import scipy as sp; print(" ".join(["-l"+lib for lib in sp.__config__.blas_opt_info["libraries"]]))'], check : true).stdout().strip()
#   message('SCIPY BLAS INFORMATION')
#   message(blas_incdir)
#   message(blas_libdirs)
#   message(blas_libs)
# endif 

## Assemble the linker arguments for BLAS
# blas_link_args = []
# foreach libdir : blas_libdirs.split(' ')
#   blas_link_args += '-L' + libdir
# endforeach
# blas_link_args += blas_libs.split(' ')
# message(blas_link_args)
# blas = declare_dependency(
#   link_args: blas_link_args,
#   include_directories: include_directories(blas_incdir)
# )

# scipy_lapack_found = run_command(py, ['-c', 'import scipy; scipy.__config__.CONFIG["Build Dependencies"]["lapack"]["found"]'], check : false).stdout().strip()
# scipy_lapack2_found = run_command(py, ['-c', 'import scipy; "lapack_opt_info" in dir(scipy.__config__)'], check : false).stdout().strip()

## Use SciPy-configured LAPACK 
# lapack_incdir = run_command(py, ['-c', 'import scipy as sp; print(" ".join(sp.__config__.lapack_opt_info["include_dirs"]))'], check : true).stdout().strip()
# lapack_libdirs = run_command(py, ['-c', 'import scipy as sp; print(" ".join(sp.__config__.lapack_opt_info["library_dirs"]))'], check : true).stdout().strip()
# lapack_libs = run_command(py, ['-c', 'import scipy as sp; print(" ".join(["-l"+lib for lib in sp.__config__.lapack_opt_info["libraries"]]))'], check : true).stdout().strip()
# message('SCIPY LAPACK INFORMATION')
# message(lapack_incdir)
# message(lapack_libdirs)
# message(lapack_libs)

## Assemble the linker arguments for LAPACK
# lapack_link_args = []
# foreach libdir : lapack_libdirs.split(' ')
#   lapack_link_args += '-L' + libdir
# endforeach
# lapack_link_args += lapack_libs.split(' ')
# message(lapack_link_args)

# lapack = declare_dependency(
#   link_args: lapack_link_args,
#   include_directories: include_directories(lapack_incdir)
# )

#inc_lapack = 
# TODO: add scipy/numpy fallback
# foo_dep = dependency('foo', fallback : ['foo', 'foo_dep'])

# blas_incdir = run_command(py, ['-c', 'import scipy as sp; print(sp.__config__.blas_opt_info[\'include_dirs\'])'], check : true).stdout().strip()
# inc_blas = include_directories(blas_incdir)

## see: https://github.com/mesonbuild/meson/issues/2835
# blas_options = [get_option('blas')]
# foreach blas_imp : ['openblas',  'OpenBLAS', 'blis', 'accelerate', 'mkl', 'atlas', 'netlib', 'blas', 'BLAS']
#   if blas_imp not in blas_options
#     blas_options += [blas_imp]
#   endif
# endforeach
# blas = dependency(blas_options, required: true)
# dependency_map += {'BLAS' : blas }

## Statically link pythran-openblas
# incdir_blas = run_command(py, ['-c', 'import pythran_openblas as openblas; print(openblas.include_dirs[0])'], check : true).stdout().strip()
# libdir_blas = run_command(py, ['-c', 'import pythran_openblas as openblas; print(openblas.library_dir)'], check : true).stdout().strip()
# inc_blas = include_directories(incdir_blas)
# lib_blas = static_library('openblas', objects: [libdir_blas / 'libopenblas.a'], include_directories : libdir_blas)
# blas = declare_dependency(
#   include_directories : incdir_blas,
#   link_with : lib_blas
# )
# dependency_map += {'BLAS' : blas }
# message('BLAS linking with pythran static: ' + blas.found().to_string())

## Add BLAS/LAPACK
# lapack_options = get_option('lapack') == '' ? [] : [get_option('lapack')]
# foreach lapack_imp : ['openblas',  'OpenBLAS', 'blis', 'accelerate',  'mkl', 'atlas', 'netlib', 'blas', 'BLAS']
#   if lapack_imp not in lapack_options
#     lapack_options += [lapack_imp]
#   endif
# endforeach
# lapack = dependency(lapack_options, required: true)
# dependency_map += {'LAPACK' : lapack }

# blas_preferred = 
# lapack_preferred = 
# if blas_preferred == '' 
#   blas_options += 
# else
#   blas_options += ['openblas',  'OpenBLAS', 'blis', 'accelerate',  'mkl', 'atlas', 'netlib', 'blas']
# endif 
# if lapack_preferred == '':
# end



# cblas = dependency('cblas', required: true)
#blas_linker_flags = blas.get_variable(cmake: 'blas', default_value : '-lblas')


## 
# lapack_dep2 = dependency('lapack', cmake_module_path: 'cmake_modules', required: true)
# lapack_lib = lapack_dep2.get_variable(cmake: 'LAPACK_LIBRARIES', default_value: '')
# lapack_linker_flags = lapack_dep2.get_variable(cmake: 'LAPACK_LINKER_FLAGS', default_value : '-llapack -lblas')
# message('LAPACK libraries found: ', lapack_lib)
# message('LAPACK linker flags: ', lapack_linker_flags)


# lapack_linker_flags = compiler.get_supported_arguments(lapack_linker_flags)
# blas_linker_flags = compiler.get_supported_arguments(blas_linker_flags)
# add_project_arguments(lapack_linker_flags, language : 'cpp')
# add_project_arguments(blas_linker_flags, language : 'cpp')

## BLAS
## Taken directly from SciPy: https://github.com/scipy/scipy/blob/c3ec353c503d23c1357e78ebff750755b3def8c6/scipy/meson.build#L154-L178

# blas_name = get_option('blas')
# if blas_name == 'openblas'
#   blas = dependency(['openblas', 'OpenBLAS'])
# else
#   blas = dependency(blas_name)
# endif
# if blas_name == 'blas'
#   cblas = dependency('cblas')
# else
#   cblas = []
# endif

# ## LAPACK is required 
# ## Taken directly from SciPy: https://github.com/scipy/scipy/blob/c3ec353c503d23c1357e78ebff750755b3def8c6/scipy/meson.build#L154-L178
# lapack_name = get_option('lapack')
# if lapack_name == 'openblas'
#   lapack = dependency(['openblas', 'OpenBLAS'], required: true)
# else
#   lapack = dependency(lapack_name, required: true)
# endif



## Dependencies information
# foreach name, dep : dependency_map
#   message(name + '_NAME', dep.name())
#   message(name + '_FOUND', dep.found())
#   if dep.found()
#     message(name + '_VERSION', dep.version())
#     message(name + '_TYPE_NAME', dep.type_name())
#     message(name + '_INCLUDEDIR', dep.get_variable('includedir', default_value: 'unknown'))
#     message(name + '_LIBDIR', dep.get_variable('libdir', default_value: 'unknown'))
#     message(name + '_PCFILEDIR', dep.get_variable('pcfiledir', default_value: 'unknown'))
#   endif
# endforeach

## Compile the package directory
subdir('src' / 'primate')


# py.install_sources(
#   'src/simplextree/__init__.py',
#   subdir: 'src/simplextree',
# )
# Install the package
# install_subdir('imfate', install_dir: py.get_install_dir(pure: false))