#ifndef _OMP_SUPPORT_H
#define _OMP_SUPPORT_H

#ifdef OMP_MULTITHREADED
   #include <omp.h>   // omp_set_num_threads, omp_get_thread_num
#else
   #define omp_get_thread_num() 0
	 #define omp_set_num_threads(x) 0
   #define omp_get_max_threads() 1 
#endif


#endif