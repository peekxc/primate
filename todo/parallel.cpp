
// Parallel computation for testing
void parallel_computation(py::array_t<double>& result_array, int num_threads) {
  int num_threads_ = num_threads;
  if (num_threads_ < 1) {
    num_threads_ = omp_get_max_threads();
  }
  num_threads_ = std::max(num_threads_, 1);
  omp_set_num_threads(num_threads);

  // Get a pointer to the underlying data of the NumPy array
  auto result_ptr = static_cast< double* >(result_array.request().ptr);

  // Get the size of the result array
  size_t size = result_array.size();

  // Release the GIL before parallel computation
  {
      py::gil_scoped_release gil_release;

      // Parallel computation using OpenMP
      #pragma omp parallel for
      for (size_t i = 0; i < size; ++i) {
          result_ptr[i] = i * 2.0;
      }
  }
}