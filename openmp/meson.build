use_openmp = get_option('use_openmp') and os_platform != 'windows' and host_machine.cpu() == 'x86_64'
if use_openmp
  add_global_arguments('-DOMP_MULTITHREADED=1', language : 'cpp')
  message('Compiling with OpenMP support')

	## Include OpenMP (mandatory ; but exclude on windows because it's too difficult to link) 
	omp = dependency('openmp', required: false)
	openmp_flags = []
	if omp.found()
		## Handle msvc, gcc, clang, or clang-cl
		if os_platform == 'windows' and cpp.get_id() == 'msvc'
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
