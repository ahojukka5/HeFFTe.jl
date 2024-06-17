"""
Heffte.API implements the C interface to the HeFFTe library.
"""
module API

using MPI

# Load the HeFFTe shared library
const libheffte = "/opt/heffte/2.3.0/lib/libheffte.so"

# Define constants from heffte_c_defines.h
const Heffte_BACKEND_STOCK::Cint = 0
const Heffte_BACKEND_FFTW::Cint = 1
const Heffte_BACKEND_MKL::Cint = 2
const Heffte_BACKEND_CUFFT::Cint = 10
const Heffte_BACKEND_ROCFFT::Cint = 11

const Heffte_SUCCESS::Cint = 0

const Heffte_RESHAPE_ALGORITHM_ALLTOALLV::Cint = 0
const Heffte_RESHAPE_ALGORITHM_P2P_PLINED::Cint = 1
const Heffte_RESHAPE_ALGORITHM_P2P::Cint = 2

const Heffte_SCALE_NONE::Cint = 0
const Heffte_SCALE_FULL::Cint = 1
const Heffte_SCALE_SYMMETRIC::Cint = 2

# Define types

mutable struct HefftePlanOptions
    use_reorder::Cint
    algorithm::Cint
    use_pencils::Cint
    use_gpu_aware::Cint
end

mutable struct HefftePlan
    backend_type::Cint
    using_r2c::Cint
    fft::Ptr{Cvoid}
end

# Define C function signatures

function heffte_set_default_options(backend::Cint, options::Ptr{HefftePlanOptions})
    ccall((libheffte, :heffte_set_default_options), Cint, (Cint, Ptr{HefftePlanOptions}), backend, options)
end

function heffte_plan_create(backend::Cint, inbox_low::Vector{Cint}, inbox_high::Vector{Cint}, inbox_order::Ptr{Cint}, 
    outbox_low::Vector{Cint}, outbox_high::Vector{Cint}, outbox_order::Ptr{Cint}, 
    comm::MPI.Comm, options::Ref{HefftePlanOptions}, plan::Ref{Ptr{HefftePlan}})::Cint
    ccall((:heffte_plan_create, libheffte), Cint, 
        (Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, MPI.Comm, Ref{HefftePlanOptions}, Ref{Ptr{HefftePlan}}),
        backend, inbox_low, inbox_high, inbox_order, outbox_low, outbox_high, outbox_order, comm, options, plan)
end

function heffte_plan_create_r2c(backend::Cint, inbox_low::Vector{Cint}, inbox_high::Vector{Cint}, inbox_order::Ptr{Cint}, 
                                outbox_low::Vector{Cint}, outbox_high::Vector{Cint}, outbox_order::Ptr{Cint}, 
                                r2c_direction::Cint, comm::MPI.Comm, options::Ref{HefftePlanOptions}, plan::Ref{Ptr{HefftePlan}})::Cint
    ccall((:heffte_plan_create_r2c, libheffte), Cint, 
          (Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, MPI.Comm, Ref{HefftePlanOptions}, Ref{Ptr{HefftePlan}}),
          backend, inbox_low, inbox_high, inbox_order, outbox_low, outbox_high, outbox_order, r2c_direction, comm, options, plan)
end

function heffte_plan_destroy(plan::Ptr{HefftePlan})::Cint
    ccall((:heffte_plan_destroy, libheffte), Cint, (Ptr{HefftePlan},), plan)
end

function heffte_size_inbox(plan::Ptr{HefftePlan})::Cint
    ccall((:heffte_size_inbox, libheffte), Cint, (Ptr{HefftePlan},), plan)
end

function heffte_size_outbox(plan::Ptr{HefftePlan})::Cint
    ccall((:heffte_size_outbox, libheffte), Cint, (Ptr{HefftePlan},), plan)
end

function heffte_size_workspace(plan::Ptr{HefftePlan})::Cint
    ccall((:heffte_size_workspace, libheffte), Cint, (Ptr{HefftePlan},), plan)
end

function heffte_get_backend(plan::Ptr{HefftePlan})::Cint
    ccall((:heffte_get_backend, libheffte), Cint, (Ptr{HefftePlan},), plan)
end

function heffte_is_r2c(plan::Ptr{HefftePlan})::Cint
    ccall((:heffte_is_r2c, libheffte), Cint, (Ptr{HefftePlan},), plan)
end

function heffte_forward_s2c(plan::Ptr{HefftePlan}, input::Ptr{Float32}, output::Ptr{Complex{Float32}}, scale::Cint)
    ccall((:heffte_forward_s2c, libheffte), Cvoid, (Ptr{HefftePlan}, Ptr{Float32}, Ptr{Complex{Float32}}, Cint), plan, input, output, scale)
end

function heffte_forward_c2c(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float32}}, output::Ptr{Complex{Float32}}, scale::Cint)
    ccall((:heffte_forward_c2c, libheffte), Cvoid, (Ptr{HefftePlan}, Ptr{Complex{Float32}}, Ptr{Complex{Float32}}, Cint), plan, input, output, scale)
end

function heffte_forward_d2z(plan::Ptr{HefftePlan}, input::Ptr{Float64}, output::Ptr{Complex{Float64}}, scale::Cint)
    ccall((:heffte_forward_d2z, libheffte), Cvoid, (Ptr{HefftePlan}, Ptr{Float64}, Ptr{Complex{Float64}}, Cint), plan, input, output, scale)
end

function heffte_forward_z2z(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float64}}, output::Ptr{Complex{Float64}}, scale::Cint)
    ccall((:heffte_forward_z2z, libheffte), Cvoid, (Ptr{HefftePlan}, Ptr{Complex{Float64}}, Ptr{Complex{Float64}}, Cint), plan, input, output, scale)
end

function heffte_backward_c2s(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float32}}, output::Ptr{Float32}, scale::Cint)
    ccall((:heffte_backward_c2s, libheffte), Cvoid, (Ptr{HefftePlan}, Ptr{Complex{Float32}}, Ptr{Float32}, Cint), plan, input, output, scale)
end

function heffte_backward_c2c(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float32}}, output::Ptr{Complex{Float32}}, scale::Cint)
    ccall((:heffte_backward_c2c, libheffte), Cvoid, (Ptr{HefftePlan}, Ptr{Complex{Float32}}, Ptr{Complex{Float32}}, Cint), plan, input, output, scale)
end

function heffte_backward_z2d(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float64}}, output::Ptr{Float64}, scale::Cint)
    ccall((:heffte_backward_z2d, libheffte), Cvoid, (Ptr{HefftePlan}, Ptr{Complex{Float64}}, Ptr{Float64}, Cint), plan, input, output, scale)
end

function heffte_backward_z2z(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float64}}, output::Ptr{Complex{Float64}}, scale::Cint)
    ccall((:heffte_backward_z2z, libheffte), Cvoid, (Ptr{HefftePlan}, Ptr{Complex{Float64}}, Ptr{Complex{Float64}}, Cint), plan, input, output, scale)
end

# Buffered versions of forward and backward transforms

function heffte_forward_s2c_buffered(plan::Ptr{HefftePlan}, input::Ptr{Float32}, output::Ptr{Complex{Float32}}, workspace::Ptr{Complex{Float32}}, scale::Cint)
    ccall((:heffte_forward_s2c_buffered, libheffte), Cvoid, (Ptr{HefftePlan}, Ptr{Float32}, Ptr{Complex{Float32}}, Ptr{Complex{Float32}}, Cint), plan, input, output, workspace, scale)
end

function heffte_forward_c2c_buffered(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float32}}, output::Ptr{Complex{Float32}}, workspace::Ptr{Complex{Float32}}, scale::Cint)
    ccall((:heffte_forward_c2c_buffered, libheffte), Cvoid, (Ptr{HefftePlan}, Ptr{Complex{Float32}}, Ptr{Complex{Float32}}, Ptr{Complex{Float32}}, Cint), plan, input, output, workspace, scale)
end

function heffte_forward_d2z_buffered(plan::Ptr{HefftePlan}, input::Ptr{Float64}, output::Ptr{Complex{Float64}}, workspace::Ptr{Complex{Float64}}, scale::Cint)
    ccall((:heffte_forward_d2z_buffered, libheffte), Cvoid, (Ptr{HefftePlan}, Ptr{Float64}, Ptr{Complex{Float64}}, Ptr{Complex{Float64}}, Cint), plan, input, output, workspace, scale)
end

function heffte_forward_z2z_buffered(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float64}}, output::Ptr{Complex{Float64}}, workspace::Ptr{Complex{Float64}}, scale::Cint)
    ccall((:heffte_forward_z2z_buffered, libheffte), Cvoid, (Ptr{HefftePlan}, Ptr{Complex{Float64}}, Ptr{Complex{Float64}}, Ptr{Complex{Float64}}, Cint), plan, input, output, workspace, scale)
end

function heffte_backward_c2s_buffered(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float32}}, output::Ptr{Float32}, workspace::Ptr{Complex{Float32}}, scale::Cint)
    ccall((:heffte_backward_c2s_buffered, libheffte), Cvoid, (Ptr{HefftePlan}, Ptr{Complex{Float32}}, Ptr{Float32}, Ptr{Complex{Float32}}, Cint), plan, input, output, workspace, scale)
end

function heffte_backward_c2c_buffered(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float32}}, output::Ptr{Complex{Float32}}, workspace::Ptr{Complex{Float32}}, scale::Cint)
    ccall((:heffte_backward_c2c_buffered, libheffte), Cvoid, (Ptr{HefftePlan}, Ptr{Complex{Float32}}, Ptr{Complex{Float32}}, Ptr{Complex{Float32}}, Cint), plan, input, output, workspace, scale)
end

function heffte_backward_z2d_buffered(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float64}}, output::Ptr{Float64}, workspace::Ptr{Complex{Float64}}, scale::Cint)
    ccall((:heffte_backward_z2d_buffered, libheffte), Cvoid, (Ptr{HefftePlan}, Ptr{Complex{Float64}}, Ptr{Float64}, Ptr{Complex{Float64}}, Cint), plan, input, output, workspace, scale)
end

function heffte_backward_z2z_buffered(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float64}}, output::Ptr{Complex{Float64}}, workspace::Ptr{Complex{Float64}}, scale::Cint)
    ccall((:heffte_backward_z2z_buffered, libheffte), Cvoid, (Ptr{HefftePlan}, Ptr{Complex{Float64}}, Ptr{Complex{Float64}}, Ptr{Complex{Float64}}, Cint), plan, input, output, workspace, scale)
end

end # module
