# CUDA C Programing

CUDA code for testing method and features.

## Compile

```bash
mkdir build && cd build
cmake ..
make
```

executable file path: `build/bin`


---

# Content Record

## Samples

Some cuda features and method examples. Like `Stream`, `__shared__ memory`, `Event Profiler`..

## Chapter2

### sumMatrix Compare

All kernal function run On Device : `NVIDIA GeForce RTX 3090`


`sumMatrixOnGPU-2D-grid-2D-block kernel `:
| Kernel Config | Run Time |
|:---------:|:----------:|
| (512 , 512), (32, 32) | 0.004128 s |
| (512 , 1024), (32, 16) | 0.004018 s |
| (1024 , 1024), (16, 16) | 0.004022 s |



`sumMatrixOnGPU-1D-grid-1D-block kernel `:
| Kernel Config | Run Time |
|:---------:|:----------:|
| (512, 1), (32, 1) | 0.040558 s |
| (128, 1), (128, 1) | 0.007047 s |


`sumMatrixOnGPU-2D-grid-1D-block kernel `:

| Kernel Config | Run Time |
|:---------:|:----------:|
| (512, 16384), (32, 1) | 0.023303 s |
| (64, 16384), (256, 1) | 0.004072 s |
