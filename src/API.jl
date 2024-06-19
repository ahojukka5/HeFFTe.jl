"""
Heffte.API implements the C interface to the HeFFTe library.
"""
module API

using MPI
using Libdl
using ..Heffte.Preferences: heffte_path

function get_libheffte()

    # if heffte_path is not set, give warning and instructions that path should be
    # set by user in order to make this package work
    if heffte_path == nothing
        println("HeFFTe path not set; please set it using `Heffte.Preferences.set_heffte_path(\"path/to/heffte\")`")
        println("Without settings path, HeFFTe.jl will not work.")
        println("After settings the path, restart your Julia session for the change to take effect.")
        return nothing
    end

    libheffte = joinpath(heffte_path, "lib", "libheffte.so")
    if !isfile(libheffte)
        throw(ArgumentError("libheffte.so not found in $(new_path)/lib"))
    end

    return libheffte
end

# Load the HeFFTe shared library
const libheffte = get_libheffte()

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

# Unsafe converts

Base.unsafe_convert(::Type{Ptr{HefftePlan}}, plan::HefftePlan) = Ptr{HefftePlan}(pointer_from_objref(plan))
Base.unsafe_convert(::Type{Ptr{HefftePlanOptions}}, plan_options::HefftePlanOptions) = Ptr{HefftePlanOptions}(pointer_from_objref(plan_options))

# Define C function signatures

function heffte_set_default_options(backend::Integer, options::HefftePlanOptions)
    @ccall libheffte.heffte_set_default_options(backend::Cint, options::Ptr{HefftePlanOptions})::Cint
end

function heffte_plan_create(backend::Integer,
                            inbox_low::Vector{T}, inbox_high::Vector{T}, inbox_order::Union{Vector{T}, Nothing},
                            outbox_low::Vector{T}, outbox_high::Vector{T}, outbox_order::Union{Vector{T}, Nothing},
                            comm::MPI.Comm, options::HefftePlanOptions, plan::Ref{Ptr{HefftePlan}}) where {T <: Integer}
    inbox_low_ = trunc.(Cint, inbox_low)
    inbox_high_ = trunc.(Cint, inbox_high)
    inbox_order_ = (inbox_order == nothing) ? Ptr{Cint}(C_NULL) : trunc.(Cint, inbox_order)
    outbox_low_ = trunc.(Cint, outbox_low)
    outbox_high_ = trunc.(Cint, outbox_high)
    outbox_order_ = (outbox_order == nothing) ? Ptr{Cint}(C_NULL) : trunc.(Cint, outbox_order)
    @ccall libheffte.heffte_plan_create(backend::Cint, inbox_low_::Ptr{Cint}, inbox_high_::Ptr{Cint}, inbox_order_::Ptr{Cint},
                                        outbox_low_::Ptr{Cint}, outbox_high_::Ptr{Cint}, outbox_order_::Ptr{Cint},
                                        comm::MPI.Comm, options::Ref{HefftePlanOptions}, plan::Ref{Ptr{HefftePlan}})::Cint
end

function heffte_plan_create_r2c(backend::Integer,
                                inbox_low::Vector{T}, inbox_high::Vector{T}, inbox_order::Union{Vector{T}, Nothing},
                                outbox_low::Vector{T}, outbox_high::Vector{T}, outbox_order::Union{Vector{T}, Nothing},
                                r2c_direction::Integer, comm::MPI.Comm, options::HefftePlanOptions, plan::Ref{Ptr{HefftePlan}}) where {T <: Integer}
    inbox_low_ = trunc.(Cint, inbox_low)
    inbox_high_ = trunc.(Cint, inbox_high)
    inbox_order_ = (inbox_order == nothing) ? Ptr{Cint}(C_NULL) : trunc.(Cint, inbox_order)
    outbox_low_ = trunc.(Cint, outbox_low)
    outbox_high_ = trunc.(Cint, outbox_high)
    outbox_order_ = (outbox_order == nothing) ? Ptr{Cint}(C_NULL) : trunc.(Cint, outbox_order)
    @ccall libheffte.heffte_plan_create_r2c(backend::Cint, inbox_low_::Ptr{Cint}, inbox_high_::Ptr{Cint}, inbox_order_::Ptr{Cint},
                                            outbox_low_::Ptr{Cint}, outbox_high_::Ptr{Cint}, outbox_order_::Ptr{Cint},
                                            r2c_direction::Cint, comm::MPI.Comm, options::Ref{HefftePlanOptions}, plan::Ref{Ptr{HefftePlan}})::Cint
end

function heffte_plan_destroy(plan::HefftePlan)
    @ccall libheffte.heffte_plan_destroy(plan::Ptr{HefftePlan})::Cint
end

function heffte_size_inbox(plan::HefftePlan)
    @ccall libheffte.heffte_size_inbox(plan::Ptr{HefftePlan})::Cint
end

function heffte_size_outbox(plan::HefftePlan)
    @ccall libheffte.heffte_size_outbox(plan::Ptr{HefftePlan})::Cint
end

function heffte_size_workspace(plan::HefftePlan)
    @ccall libheffte.heffte_size_workspace(plan::Ptr{HefftePlan})::Cint
end

function heffte_get_backend(plan::HefftePlan)
    @ccall libheffte.heffte_get_backend(plan::Ptr{HefftePlan})::Cint
end

function heffte_is_r2c(plan::HefftePlan)
    @ccall libheffte.heffte_is_r2c(plan::Ptr{HefftePlan})::Cint
end

function heffte_forward_s2c(plan::HefftePlan, input::Vector{Float32}, output::Vector{Complex{Float32}}, scale::Integer)
    @ccall libheffte.heffte_forward_s2c(plan::Ptr{HefftePlan}, input::Ptr{Float32}, output::Ptr{Complex{Float32}}, scale::Cint)::Cint
end

function heffte_forward_c2c(plan::HefftePlan, input::Vector{Complex{Float32}}, output::Vector{Complex{Float32}}, scale::Integer)
    @ccall libheffte.heffte_forward_c2c(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float32}}, output::Ptr{Complex{Float32}}, scale::Cint)::Cint
end

function heffte_forward_d2z(plan::HefftePlan, input::Vector{Float64}, output::Vector{Complex{Float64}}, scale::Integer)
    @ccall libheffte.heffte_forward_d2z(plan::Ptr{HefftePlan}, input::Ptr{Float64}, output::Ptr{Complex{Float64}}, scale::Cint)::Cint
end

function heffte_forward_z2z(plan::HefftePlan, input::Vector{Complex{Float64}}, output::Vector{Complex{Float64}}, scale::Integer)
    @ccall libheffte.heffte_forward_z2z(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float64}}, output::Ptr{Complex{Float64}}, scale::Cint)::Cint
end

function heffte_backward_c2s(plan::HefftePlan, input::Vector{Complex{Float32}}, output::Vector{Float32}, scale::Integer)
    @ccall libheffte.heffte_backward_c2s(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float32}}, output::Ptr{Float32}, scale::Cint)::Cint
end

function heffte_backward_c2c(plan::HefftePlan, input::Vector{Complex{Float32}}, output::Vector{Complex{Float32}}, scale::Integer)
    @ccall libheffte.heffte_backward_c2c(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float32}}, output::Ptr{Complex{Float32}}, scale::Cint)::Cint
end

function heffte_backward_z2d(plan::HefftePlan, input::Vector{Complex{Float64}}, output::Vector{Float64}, scale::Integer)
    @ccall libheffte.heffte_backward_z2d(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float64}}, output::Ptr{Float64}, scale::Cint)::Cint
end

function heffte_backward_z2z(plan::HefftePlan, input::Vector{Complex{Float64}}, output::Vector{Complex{Float64}}, scale::Integer)
    @ccall libheffte.heffte_backward_z2z(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float64}}, output::Ptr{Complex{Float64}}, scale::Cint)::Cint
end

# Buffered versions of forward and backward transforms

function heffte_forward_s2c_buffered(plan::HefftePlan, input::Vector{Float32}, output::Vector{Complex{Float32}}, workspace::Vector{Complex{Float32}}, scale::Integer)
    @ccall libheffte.heffte_forward_s2c_buffered(plan::Ptr{HefftePlan}, input::Ptr{Float32}, output::Ptr{Complex{Float32}}, workspace::Ptr{Complex{Float32}}, scale::Cint)::Cint
end

function heffte_forward_c2c_buffered(plan::HefftePlan, input::Vector{Complex{Float32}}, output::Vector{Complex{Float32}}, workspace::Vector{Complex{Float32}}, scale::Integer)
    @ccall libheffte.heffte_forward_c2c_buffered(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float32}}, output::Ptr{Complex{Float32}}, workspace::Ptr{Complex{Float32}}, scale::Cint)::Cint
end

function heffte_forward_d2z_buffered(plan::HefftePlan, input::Vector{Float64}, output::Vector{Complex{Float64}}, workspace::Vector{Complex{Float64}}, scale::Integer)
    @ccall libheffte.heffte_forward_d2z_buffered(plan::Ptr{HefftePlan}, input::Ptr{Float64}, output::Ptr{Complex{Float64}}, workspace::Ptr{Complex{Float64}}, scale::Cint)::Cint
end

function heffte_forward_z2z_buffered(plan::HefftePlan, input::Vector{Complex{Float64}}, output::Vector{Complex{Float64}}, workspace::Vector{Complex{Float64}}, scale::Integer)
    @ccall libheffte.heffte_forward_z2z_buffered(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float64}}, output::Ptr{Complex{Float64}}, workspace::Ptr{Complex{Float64}}, scale::Cint)::Cint
end

function heffte_backward_c2s_buffered(plan::HefftePlan, input::Vector{Complex{Float32}}, output::Vector{Float32}, workspace::Vector{Complex{Float32}}, scale::Integer)
    @ccall libheffte.heffte_backward_c2s_buffered(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float32}}, output::Ptr{Float32}, workspace::Ptr{Complex{Float32}}, scale::Cint)::Cint
end

function heffte_backward_c2c_buffered(plan::HefftePlan, input::Vector{Complex{Float32}}, output::Vector{Complex{Float32}}, workspace::Vector{Complex{Float32}}, scale::Integer)
    @ccall libheffte.heffte_backward_c2c_buffered(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float32}}, output::Ptr{Complex{Float32}}, workspace::Ptr{Complex{Float32}}, scale::Cint)::Cint
end

function heffte_backward_z2d_buffered(plan::HefftePlan, input::Vector{Complex{Float64}}, output::Vector{Float64}, workspace::Vector{Complex{Float64}}, scale::Integer)
    @ccall libheffte.heffte_backward_z2d_buffered(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float64}}, output::Ptr{Float64}, workspace::Ptr{Complex{Float64}}, scale::Cint)::Cint
end

function heffte_backward_z2z_buffered(plan::HefftePlan, input::Vector{Complex{Float64}}, output::Vector{Complex{Float64}}, workspace::Vector{Complex{Float64}}, scale::Integer)
    @ccall libheffte.heffte_backward_z2z_buffered(plan::Ptr{HefftePlan}, input::Ptr{Complex{Float64}}, output::Ptr{Complex{Float64}}, workspace::Ptr{Complex{Float64}}, scale::Cint)::Cint
end

# export all

export Heffte_BACKEND_STOCK, Heffte_BACKEND_FFTW, Heffte_BACKEND_MKL, Heffte_BACKEND_CUFFT, Heffte_BACKEND_ROCFFT
export Heffte_SUCCESS
export Heffte_RESHAPE_ALGORITHM_ALLTOALLV, Heffte_RESHAPE_ALGORITHM_P2P_PLINED, Heffte_RESHAPE_ALGORITHM_P2P
export Heffte_SCALE_NONE, Heffte_SCALE_FULL, Heffte_SCALE_SYMMETRIC
export HefftePlanOptions, HefftePlan
export heffte_set_default_options, heffte_plan_create, heffte_plan_create_r2c, heffte_plan_destroy
export heffte_size_inbox, heffte_size_outbox, heffte_size_workspace
export heffte_get_backend, heffte_is_r2c
export heffte_forward_s2c, heffte_forward_c2c, heffte_forward_d2z, heffte_forward_z2z
export heffte_backward_c2s, heffte_backward_c2c, heffte_backward_z2d, heffte_backward_z2z
export heffte_forward_s2c_buffered, heffte_forward_c2c_buffered, heffte_forward_d2z_buffered, heffte_forward_z2z_buffered
export heffte_backward_c2s_buffered, heffte_backward_c2c_buffered, heffte_backward_z2d_buffered, heffte_backward_z2z_buffered

end # module
