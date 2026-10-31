# Heffte.jl

Julia bindings for [HeFFTe](https://github.com/icl-utk-edu/heffte). The shared
library is [`HeFFTe_jll`](https://github.com/JuliaBinaryWrappers/HeFFTe_jll.jl)
v2.4.1.

`HeFFTe_jll` v2.4.1+0 publishes `libheffte` for Linux x86_64 and aarch64.
macOS binaries are not in that release. `Box3D`, `proc_setup_min_surface` and
`split_world` do not call the native library. `Plan`, `forward!` and
`backward!` do.

`Heffte.API` wraps the C functions in `heffte_c.h`. The high-level plan is one
real-to-complex FFT. Complex-to-complex and real-to-complex plans are still
separate work.
