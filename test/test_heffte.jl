# Buffered high-level roundtrip. Requires one or more MPI ranks and libheffte.
# Not part of ordinary Pkg.test().

using Heffte
using MPI
using Test

MPI.Init()
comm = MPI.COMM_WORLD

try
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # 0-based 4³ world. A constant field has one unnormalized bin of magnitude N.
    world = Heffte.Box3D((0, 0, 0), (3, 3, 3))
    n_global = Heffte.count(world)
    grid = Heffte.proc_setup_min_surface(world, nranks)
    boxes = Heffte.split_world(world, grid)
    mine = boxes[rank + 1]

    options = Heffte.PlanOptions(:FFTW)
    plan = Heffte.Plan(mine, mine, options)

    input = ones(Heffte.size_inbox(plan))
    output = zeros(Complex{Float64}, Heffte.size_outbox(plan))
    workspace = zeros(Complex{Float64}, Heffte.size_workspace(plan))

    # Wrapper defaults: forward :none, backward :full. That pair restores the field.
    Heffte.forward!(plan, input, output, workspace)
    local_length = length(input)
    total_length = MPI.Allreduce(local_length, +, comm)
    local_sum = sum(output)
    spectral_sum = MPI.Allreduce(real(local_sum), +, comm) +
                   im * MPI.Allreduce(imag(local_sum), +, comm)

    restored = zeros(Float64, Heffte.size_inbox(plan))
    Heffte.backward!(plan, output, restored, workspace)

    @testset "buffered FFT roundtrip" begin
        @test total_length == n_global
        @test all(isfinite, output)
        @test all(isfinite, restored)
        # Unnormalized transform of the constant 1 sums to N. A copy or a
        # zero transform does not.
        @test spectral_sum ≈ n_global + 0im atol=1e-8 * n_global
        @test restored ≈ input atol=1e-10
    end
finally
    MPI.Finalize()
end
