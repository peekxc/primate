# Adds BLAS / LAPACK support to the current build 
allow_noblas = false 

# Platform detection
is_windows = host_machine.system() == 'windows'
is_mingw = is_windows and cc.get_id() == 'gcc'

# For mingw-w64, link statically against the UCRT.
if is_mingw
  gcc_link_args = ['-lucrt', '-static']
  add_project_link_arguments(gcc_link_args, language: ['cpp'])
endif

# Configure the BLAS options
blas_name = get_option('blas')
lapack_name = get_option('lapack')
allow_noblas = get_option('allow-noblas')
blas_symbol_suffix = get_option('blas-symbol-suffix')
use_ilp64 = get_option('use-ilp64')
blas_interface = use_ilp64 ? ['interface: ilp64'] : ['interface: lp64']

## Write a pkgconfig file (*.pc) for scipy's openblas
# r = run_command('python', 'write_openblas_pc.py', check : true)
# if r.returncode() != 0
#   error('Failed to setup openblas pkgconfig file.')
# endif
# pc_dir = r.stdout().strip()
# message(pc_dir)
# env.prepend('PKG_CONFIG_PATH', pc_dir)

# # Try to link to SciPy's openblas 
# if blas_name == 'openblas' or blas_name == 'auto'
#   blas = dependency('scipy-openblas', required: false, method : 'pkg-config')
#   if blas.found()
#     blas_name = 'scipy-openblas'
#   endif
# endif
# # message(run_command('pkg-config', '--libs', 'scipy-openblas', check: true).stdout())
# message(blas_name)

# if blas_name == 'openblas' or blas_name == 'auto'
#   message('Using SciPy openblas')
#   incdir_blas = run_command(py, ['-c', 'import scipy_openblas64; print(scipy_openblas64.get_include_dir())'], check : true).stdout().strip()
#   # inc_blas = include_directories(incdir_blas)
#   libdir_blas = run_command(py, ['-c', 'import scipy_openblas64; print(scipy_openblas64.get_lib_dir())'], check : true).stdout().strip()
#   libfn_blas = run_command(py, ['-c', 'import scipy_openblas64; print(scipy_openblas64.get_library())'], check : true).stdout().strip()
#   blas = declare_dependency(
#     link_args : ['-L'+libdir_blas, '-l'+libfn_blas],
#     include_directories : include_directories(incdir_blas)
#   )
#   _cpp_args += '-L'+libdir_blas
#   _cpp_args += '-l'+libfn_blas
#   _cpp_args += '-lopenblas'
#   message(blas.found())
#   # blas = dependency('scipy-openblas', required: false, method : 'pkg-config')
#   blas_name = 'scipy-openblas' 
# endif 

# Go through the order of implementations to try  
if blas_name == 'auto'
  message(blas_name)
  message('BLAS name is apparently equal to auto so we have to detect everything?')
  foreach _name : get_option('blas-order')
    if _name == 'mkl'
      blas = dependency('mkl',
        modules: ['cblas'] + blas_interface + mkl_opts,
        required: false,  # may be required, but we need to emit a custom error message
        version: mkl_version_req,
      )
      # Insert a second try with MKL, because we may be rejecting older versions
      # or missing it because no pkg-config installed. If so, we need to retry
      # with MKL SDL, and drop the version constraint (this always worked).
      if not blas.found() and mkl_may_use_sdl
        blas = dependency('mkl', modules: ['cblas', 'sdl: true'], required: false)
      endif
    else
      if _name == 'flexiblas' and use_ilp64
        _name = 'flexiblas64'
      endif
      blas = dependency(_name, modules: ['cblas'] + blas_interface, required: false)
    endif
    if blas.found()
      break
    endif
  endforeach
else
  if blas_name == 'mkl'
    blas = dependency('mkl',
      modules: ['cblas'] + blas_interface + mkl_opts,
      required: false,
      version: mkl_version_req,
    )
    # Same deal as above - try again for MKL
    if not blas.found() and mkl_may_use_sdl
      blas = dependency('mkl', modules: ['cblas', 'sdl: true'], required: false)
    endif
  else
    if blas_name != 'scipy-openblas'
      blas = dependency(blas_name, modules: ['cblas'] + blas_interface, required: false)
    endif
  endif
endif

# Detect if BLAS was found 
have_blas = blas.found()
if have_blas
  _args_blas = ['-DHAVE_CBLAS']  # note: used for C and C++ via `blas_dep` below
  if blas_symbol_suffix == 'auto'
    if blas_name == 'scipy-openblas'
      blas_symbol_suffix = '32_' # 
    else
      blas_symbol_suffix = blas.get_variable('symbol_suffix', default_value: '')
    endif
    message(f'BLAS symbol suffix: @blas_symbol_suffix@')
  endif
  if blas_symbol_suffix != ''
    _args_blas += ['-DBLAS_SYMBOL_SUFFIX=' + blas_symbol_suffix]
  endif
  blas_dep = declare_dependency(
    dependencies: [blas],
    compile_args: _args_blas,
  )
else
  if allow_noblas
    blas_dep = []
  else
    error('No BLAS library detected! Install one, or use the ' + \
          '`allow-noblas` build option (note, this may be up to 100x slower ' + \
          'for some linear algebra operations).')
  endif
endif

if 'mkl' in blas.name() or blas.name() == 'accelerate' or blas_name == 'scipy-openblas'
  # For these libraries we know that they contain LAPACK, and it's desirable to
  # use that - no need to run the full detection twice.
  lapack = blas
else
  if lapack_name == 'auto'
    foreach _name : get_option('lapack-order')
      lapack = dependency(_name, modules: ['lapack'] + blas_interface, required: false)
      if lapack.found()
        break
      endif
    endforeach
  else
    lapack = dependency(lapack_name, modules: ['lapack'] + blas_interface, required: false)
  endif
endif

have_lapack = lapack.found()
if not have_lapack
  error('No LAPACK library detected! Install one, or use the ' + \
        '`allow-noblas` build option (note, this may be up to 100x slower ' + \
        'for some linear algebra operations).')
endif

# BLAS/LAPACK dependency info. 
dependency_map = { 'LAPACK': lapack }
if have_blas
  dependency_map += { 'BLAS': blas }
endif 

# conf_data = configuration_data()
foreach name, dep : dependency_map
  message(name + '_NAME', dep.name())
  message(name + '_FOUND', dep.found())
  if dep.found()
    message(name + '_VERSION', dep.version())
    message(name + '_TYPE_NAME', dep.type_name())
    # get_variable() results may be missing for a variety of reasons
    message(name + '_INCLUDEDIR', dep.get_variable('includedir', default_value: 'unknown'))
    message(name + '_LIBDIR', dep.get_variable('libdir', default_value: 'unknown'))
    message(name + '_OPENBLAS_CONFIG', dep.get_variable('openblas_config', default_value: 'unknown'))
    message(name + '_PCFILEDIR', dep.get_variable('pcfiledir', default_value: 'unknown'))
  endif
endforeach
_cpp_args += compiler.get_supported_arguments('-llapack')
_cpp_args += compiler.get_supported_arguments('-lblas')



# lapack_linker_flags = compiler.get_supported_arguments(lapack_linker_flags)
