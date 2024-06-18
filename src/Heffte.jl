module Heffte

using Base.Iterators: product

include("API.jl")

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

Split the given `world` into subdomains based on the `proc_grid` and `remap` parameters.

## Arguments
- `world::Box3D`: The 3D box representing the world to be split.
- `proc_grid::NTuple{3, Int}`: The 3D grid specifying the number of processors in each dimension.
- `remap::Vector{Int}` (optional): A vector specifying the remapping of subdomains. Default is an empty vector.

## Returns
- An array of `Box3D` objects representing the subdomains.

The `split_world` function divides the `world` into subdomains based on the number of processors specified by `proc_grid`.
Each subdomain is represented by a `Box3D` object, which is a 3D box defined by its lower and upper coordinates.
If the `remap` vector is provided, the function remaps the subdomains according to the indices in the `remap` vector.

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

end
