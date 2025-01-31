num <- 1L
Sys.setenv(OMP_NUM_THREADS = num)
Sys.setenv(OMP_THREAD_LIMIT = num)
# Unsure, MKL is an Intel-specific thing
Sys.setenv(MKL_NUM_THREADS = num)
Sys.setenv(MC_CORES = num)
# Package-specific settings
try(data.table::setDTthreads(num))
try(RhpcBLASctl::blas_set_num_threads(num))
try(RhpcBLASctl::omp_set_num_threads(num))

# Disable GPU usage
Sys.setenv(CUDA_VISIBLE_DEVICES = "")
