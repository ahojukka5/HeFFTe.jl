module Heffte

using MPI
using Base.Iterators: product

include("Preferences.jl")
include("API.jl")

using .API

"""
    struct Box3D

A struct representing a 3D box.

# Fields
- `low::NTuple{3, Int}`: The lower corner of the box.
- `high::NTuple{3, Int}`: The upper corner of the box.
- `size::NTuple{3, Int}`: The size of the box in each dimension.
- `order::Int`: The order of the box.

"""
struct Box3D
    low::NTuple{3, Int}
    high::NTuple{3, Int}
    size::NTuple{3, Int}
    order::Int
end

"""
    Box3D(low::NTuple{3, Int}, high::NTuple{3, Int})

Constructs a 3D box with the given lower and upper bounds.

# Arguments
- `low::NTuple{3, Int}`: The lower bounds of the box.
- `high::NTuple{3, Int}`: The upper bounds of the box.

# Returns
- `Box3D`: The constructed 3D box.

# Examples
```julia
box = Box3D((1, 2, 3), (4, 5, 6))
```
"""
function Box3D(low::NTuple{3, Int}, high::NTuple{3, Int})
    size = (high[1] - low[1] + 1, high[2] - low[2] + 1, high[3] - low[3] + 1)
    return Box3D(low, high, size, 0)
end

function Box3D(high::NTuple{3, Int})
    return Box3D((1, 1, 1), high)
end

"""
    count(box::Box3D)

Compute the total number of elements in a 3D box.

# Arguments
- `box::Box3D`: The 3D box for which to compute the count.

# Returns
The total number of elements in the box.

"""
function count(box::Box3D)
    return box.size[1] * box.size[2] * box.size[3]
end

"""
    proc_setup_min_surface(world::Box3D, num_procs::Int)

Given a `world` box and the number of processors `num_procs`, this function
calculates the optimal processor grid dimensions that minimize the surface area
of each processor's subdomain. The `world` box must not be empty.

# Arguments
- `world::Box3D`: The world box representing the global domain.
- `num_procs::Int`: The number of processors.

# Returns
- `best_grid::Tuple{Int, Int, Int}`: The optimal processor grid dimensions that
  minimize the surface area.

# Example

```julia
world_box = Box3D((100, 200, 300))
num_procs = 16
grid = proc_setup_min_surface(world_box, num_procs)
println("Optimal grid dimensions: ", grid)
```
"""
function proc_setup_min_surface(world::Box3D, num_procs::Int)

    @assert count(world) > 0 "The world box must not be empty"

    all_indexes = world.size

    best_grid = (1, 1, 1)

    function surface(proc_grid)
        box_size = Tuple(a÷b for (a, b) in zip(all_indexes, proc_grid))
        return 2 * (box_size[1]*box_size[2] + box_size[1]*box_size[3] + box_size[2]*box_size[3])
    end

    best_surface = typemax(Int)

    for i in 1:min(num_procs, all_indexes[1])
        if num_procs % i == 0
            remainder = num_procs ÷ i
            for j in 1:min(remainder, all_indexes[2])
                if remainder % j == 0
                    k = remainder ÷ j
                    if k <= all_indexes[3]
                        candidate_grid = (i, j, k)
                        candidate_surface = surface(candidate_grid)
                        if candidate_surface < best_surface
                            best_surface = candidate_surface
                            best_grid = candidate_grid
                        end
                    end
                end
            end
        end
    end

    @assert prod(best_grid) == num_procs "The product of the best grid dimensions must equal num_procs"

    return best_grid
end

"""
    split_world(world::Box3D, proc_grid::NTuple{3, Int}, remap::Vector{Int}=Int[])

Split the given `world` into subdomains based on the `proc_grid` and `remap`
parameters.

## Arguments
- `world::Box3D`: The 3D box representing the world to be split.
- `proc_grid::NTuple{3, Int}`: The 3D grid specifying the number of processors
  in each dimension.
- `remap::Vector{Int}` (optional): A vector specifying the remapping of
  subdomains. Default is an empty vector.

## Returns
- An array of `Box3D` objects representing the subdomains.

The `split_world` function divides the `world` into subdomains based on the
number of processors specified by `proc_grid`. Each subdomain is represented by
a `Box3D` object, which is a 3D box defined by its lower and upper coordinates.
If the `remap` vector is provided, the function remaps the subdomains according
to the indices in the `remap` vector.

The function returns an array of `Box3D` objects representing the subdomains.
"""
function split_world(world::Box3D, proc_grid::NTuple{3, Int}, remap::Vector{Int}=Int[])

    fast = (i) -> world.low[1] + i * (world.size[1] ÷ proc_grid[1]) + min(i, (world.size[1] % proc_grid[1]))
    mid  = (i) -> world.low[2] + i * (world.size[2] ÷ proc_grid[2]) + min(i, (world.size[2] % proc_grid[2]))
    slow = (i) -> world.low[3] + i * (world.size[3] ÷ proc_grid[3]) + min(i, (world.size[3] % proc_grid[3]))

    result = []

    for k in 0:(proc_grid[3]-1)
        for j in 0:(proc_grid[2]-1)
            for i in 0:(proc_grid[1]-1)
                low = (fast(i), mid(j), slow(k))
                high = (fast(i+1)-1, mid(j+1)-1, slow(k+1)-1)
                push!(result, Box3D(low, high))
            end
        end
    end

    if isempty(remap)
        return result
    else
        remapped_result = []
        for i in 1:length(remap)
            if remap[i] == -1
                push!(remapped_result, Box3D((0, 0, 0), (-1, -1, -1)))
            else
                push!(remapped_result, result[remap[i]+1])
            end
        end
        return remapped_result
    end
end

function get_backend_type(backend::Symbol)::Int
    if backend == :STOCK
        return Heffte_BACKEND_STOCK
    elseif backend == :FFTW
        return Heffte_BACKEND_FFTW
    elseif backend == :MKL
        return Heffte_BACKEND_MKL
    elseif backend == :CUDA
        return Heffte_BACKEND_CUDA
    elseif backend == :ROCFFT
        return Heffte_BACKEND_ROCFFT
    else
        error("Unknown backend: $backend")
    end
end

"""
    PlanOptions(backend::Symbol)

Return default plan options for backend.

Choices: :STOCK, :FFTW, :MKL, :CUDA, :ROCFFT
"""
function PlanOptions(backend::Symbol)
    options = HefftePlanOptions(0, 0, 0, 0)
    backend_type = get_backend_type(backend)
    heffte_set_default_options(backend_type, options)
    return options
end

# nice formatted output for PlanOptions
function Base.show(io::IO, options::HefftePlanOptions)
    is_reorder = options.use_reorder == 1
    if options.algorithm == Heffte_RESHAPE_ALGORITHM_ALLTOALLV
        reshape_algorithm = "alltoallv"
    elseif options.algorithm == Heffte_RESHAPE_ALGORITHM_P2P_PLINED
        reshape_algorithm = "point-to-point (pipelined)"
    elseif options.algorithm == Heffte_RESHAPE_ALGORITHM_P2P
        reshape_algorithm = "point-to-point"
    end
    pencils_or_slabs = options.use_pencils == 1 ? "pencils" : "slabs"
    is_gpu_aware = options.use_gpu_aware == 1

    println(io, "HefftePlanOptions(")
    println(io, "    reorder = ", is_reorder)
    println(io, "    reshape_algorithm = ", reshape_algorithm)
    println(io, "    pencils / slabs = ", pencils_or_slabs)
    println(io, "    gpu_aware = ", is_gpu_aware)
    println(io, "    )")
end

function Plan(inbox::Box3D, outbox::Box3D, options::HefftePlanOptions; backend=:FFTW, comm=MPI.COMM_WORLD)
    plan_ref = Ref{Ptr{HefftePlan}}(C_NULL)
    backend_type = get_backend_type(backend)
    status = heffte_plan_create(backend_type, collect(inbox.low), collect(inbox.high), nothing,
                                collect(outbox.low), collect(outbox.high), nothing,
                                comm, options, plan_ref)
    if status != Heffte_SUCCESS
        error("Failed at heffte_plan_create() with error code: $status")
    end
    return unsafe_load(plan_ref[])
end

function get_backend_name(backend::Integer)::Symbol
    if backend == Heffte_BACKEND_STOCK
        return :STOCK
    elseif backend == Heffte_BACKEND_FFTW
        return :FFTW
    elseif backend == Heffte_BACKEND_MKL
        return :MKL
    elseif backend == Heffte_BACKEND_CUDA
        return :CUDA
    elseif backend == Heffte_BACKEND_ROCFFT
        return :ROCFFT
    else
        error("Unknown backend: $backend")
    end
end

# nice formatted output for Plan
function Base.show(io::IO, plan::HefftePlan)
    backend = get_backend_name(plan.backend_type)
    using_r2c = plan.using_r2c == 1
    println(io, "HefftePlan(")
    println(io, "    backend = ", backend)
    println(io, "    using_r2c = ", using_r2c)
    println(io, "    )")
end

size_inbox(plan::HefftePlan) = heffte_size_inbox(plan)
size_outbox(plan::HefftePlan) = heffte_size_outbox(plan)
size_workspace(plan::HefftePlan) = heffte_size_workspace(plan)

function get_scale(scale::Symbol)::Int
    if scale == :none
        return Heffte_SCALE_NONE
    elseif scale == :full
        return Heffte_SCALE_FULL
    elseif scale == :symmetric
        return Heffte_SCALE_SYMMETRIC
    else
        error("Unknown scale: $scale")
    end
end

function forward!(plan::HefftePlan, input::Vector{Float64}, output::Vector{Complex{Float64}}, workspace::Vector{Complex{Float64}}; scale=:none)
    heffte_forward_d2z_buffered(plan, input, output, workspace, get_scale(scale))
    return output
end

function backward!(plan::HefftePlan, input::Vector{Complex{Float64}}, output::Vector{Float64}, workspace::Vector{Complex{Float64}}; scale=:full)
    heffte_backward_z2d_buffered(plan, input, output, workspace, get_scale(scale))
    return output
end

end # module
