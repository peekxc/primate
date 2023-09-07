project(
  'pyimate', 'cpp', 
  license: 'Apache-2.0',
  meson_version: '>= 1.1.0',
  default_options : [
    'buildtype=debugoptimized', 
    'cpp_std=c++20', 
    'warning_level=2', 
    # 'fortran_std=legacy',
    # 'blas=openblas',
    # 'lapack=openblas', 
    'pkgconfig.relocatable=true'
  ]
)

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
  '-std=c++20'
)

## Include directories
incdir_numpy = run_command(py, ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'], check : true).stdout().strip()
incdir_imate = run_command(py, ['-c', 'import os; os.chdir(".."); import imate; print(imate.__path__[0])'], check : true).stdout().strip()

## Header includes
inc_np = include_directories(incdir_numpy)
inc_pybind11 = include_directories('extern' / 'pybind11' / 'include')
inc_eigen = include_directories('extern' / 'eigen')
inc_imate = include_directories(incdir_imate)
inc_pkg = include_directories('include')

## Add dependencies, if they are available
dependency_map = {}

## Include OpenMP, if possible
omp = dependency('openmp', required: false)
if omp.found()
  # _cpp_args += compiler.get_supported_arguments('-fopenmp')
  openmp_flags = compiler.get_supported_arguments('-fopenmp')
  add_project_link_arguments(openmp_flags, language : 'cpp')
  dependency_map += { 'OpenMP' : omp }
elif compiler.has_argument('-fopenmp')
  _c = compiler.has_argument('-fopenmp')
  _h = compiler.has_header('omp.h')
  if (_c and _h)
    omp = declare_dependency(compile_args:'-fopenmp', link_args:'-fopenmp')
    dependency_map += { 'OpenMP' : omp }
  endif
endif


## Set USE_CBLAS to 0 to use C++ implemented sparse matrix vector operations
## Otherwise BLAS must be linked and used
env = environment()
env.set('USE_CBLAS', '0') ## Turn off CBLAS

## Use SciPy-configured BLAS
blas_incdir = run_command(py, ['-c', 'import scipy as sp; print(" ".join(sp.__config__.blas_opt_info["include_dirs"]))'], check : true).stdout().strip()
blas_libdirs = run_command(py, ['-c', 'import scipy as sp; print(" ".join(sp.__config__.blas_opt_info["library_dirs"]))'], check : true).stdout().strip()
blas_libs = run_command(py, ['-c', 'import scipy as sp; print(" ".join(["-l"+lib for lib in sp.__config__.blas_opt_info["libraries"]]))'], check : true).stdout().strip()
message('SCIPY BLAS INFORMATION')
message(blas_incdir)
message(blas_libdirs)
message(blas_libs)

## Assemble the linker arguments for BLAS
blas_link_args = []
foreach libdir : blas_libdirs.split(' ')
  blas_link_args += '-L' + libdir
endforeach
blas_link_args += blas_libs.split(' ')
message(blas_link_args)

blas = declare_dependency(
  link_args: blas_link_args,
  include_directories: include_directories(blas_incdir)
)

## Use SciPy-configured LAPACK 
lapack_incdir = run_command(py, ['-c', 'import scipy as sp; print(" ".join(sp.__config__.lapack_opt_info["include_dirs"]))'], check : true).stdout().strip()
lapack_libdirs = run_command(py, ['-c', 'import scipy as sp; print(" ".join(sp.__config__.lapack_opt_info["library_dirs"]))'], check : true).stdout().strip()
lapack_libs = run_command(py, ['-c', 'import scipy as sp; print(" ".join(["-l"+lib for lib in sp.__config__.lapack_opt_info["libraries"]]))'], check : true).stdout().strip()
message('SCIPY LAPACK INFORMATION')
message(lapack_incdir)
message(lapack_libdirs)
message(lapack_libs)

## Assemble the linker arguments for LAPACK
lapack_link_args = []
foreach libdir : lapack_libdirs.split(' ')
  lapack_link_args += '-L' + libdir
endforeach
lapack_link_args += lapack_libs.split(' ')
message(lapack_link_args)

lapack = declare_dependency(
  link_args: lapack_link_args,
  include_directories: include_directories(lapack_incdir)
)

#inc_lapack = 
# TODO: add scipy/numpy fallback
# foo_dep = dependency('foo', fallback : ['foo', 'foo_dep'])

# blas_incdir = run_command(py, ['-c', 'import scipy as sp; print(sp.__config__.blas_opt_info[\'include_dirs\'])'], check : true).stdout().strip()
# inc_blas = include_directories(blas_incdir)

## BLAS / LAPACK
## see: https://github.com/mesonbuild/meson/issues/2835
blas_options = ['mkl', 'blis', 'openblas', 'OpenBLAS', 'accelerate', 'atlas', 'netlib', 'blas']
# lapack_options = ['mkl', 'openblas', 'libflame', 'accelerate', 'atlas', 'blas']

# blas = dependency(blas_options, required: true )
# cblas = dependency('cblas', required: true)
# lapack = dependency(lapack_options, required: true)
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

dependency_map = {
  'BLAS': blas,
  'LAPACK': lapack
}

## Dependencies information
foreach name, dep : dependency_map
  message(name + '_NAME', dep.name())
  message(name + '_FOUND', dep.found())
  if dep.found()
    message(name + '_VERSION', dep.version())
    message(name + '_TYPE_NAME', dep.type_name())
    message(name + '_INCLUDEDIR', dep.get_variable('includedir', default_value: 'unknown'))
    message(name + '_LIBDIR', dep.get_variable('libdir', default_value: 'unknown'))
    message(name + '_PCFILEDIR', dep.get_variable('pcfiledir', default_value: 'unknown'))
  endif
endforeach

## Compile the package directory
subdir('pyimate')


# py.install_sources(
#   'src/simplextree/__init__.py',
#   subdir: 'src/simplextree',
# )
# Install the package
# install_subdir('imfate', install_dir: py.get_install_dir(pure: false))