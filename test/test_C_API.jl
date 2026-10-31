# Low-level C API roundtrip on exactly two ranks. Requires libheffte.
# Not part of ordinary Pkg.test().

using Heffte.API
using MPI
using Test

MPI.Init()
comm = MPI.COMM_WORLD

try
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    if nranks != 2
        error("This script requires exactly 2 ranks")
    end

    inbox_low = Cint[0, 0, 0]
    inbox_high = Cint[3, 3, 3]
    inbox_order = Cint[0, 1, 2]
    outbox_low = Cint[0, 0, 0]
    outbox_high = Cint[3, 3, 3]
    outbox_order = Cint[0, 1, 2]

    if rank == 0
        inbox_high[3] = 1
        outbox_high[3] = 1
    else
        inbox_low[3] = 2
        outbox_low[3] = 2
    end

    backend = Heffte_BACKEND_FFTW
    options = HefftePlanOptions(
        1, Heffte_RESHAPE_ALGORITHM_ALLTOALLV, 1, 1,
    )
    plan_ref = Ref{Ptr{HefftePlan}}(C_NULL)
    status = heffte_plan_create(
        backend, inbox_low, inbox_high, inbox_order,
        outbox_low, outbox_high, outbox_order,
        comm, options, plan_ref,
    )
    plan_ptr = plan_ref[]
    if status != Heffte_SUCCESS || plan_ptr == C_NULL
        error("Failed at heffte_plan_create() with error code: $status")
    end
    plan = unsafe_load(plan_ptr)

    n_global = 4 * 4 * 4
    input = ones(heffte_size_inbox(plan))
    output = zeros(Complex{Float64}, heffte_size_outbox(plan))
    # none + full is the identity pair. Symmetric on both sides scales by 1/N.
    heffte_forward_d2z(plan, input, output, Heffte_SCALE_NONE)

    local_length = length(input)
    total_length = MPI.Allreduce(local_length, +, comm)
    local_sum = sum(output)
    spectral_sum = MPI.Allreduce(real(local_sum), +, comm) +
                   im * MPI.Allreduce(imag(local_sum), +, comm)

    restored = zeros(length(input))
    heffte_backward_z2d(plan, output, restored, Heffte_SCALE_FULL)
    heffte_plan_destroy(plan_ptr)

    @testset "C API FFT roundtrip" begin
        @test total_length == n_global
        @test all(isfinite, output)
        @test all(isfinite, restored)
        @test spectral_sum ≈ n_global + 0im atol=1e-8 * n_global
        @test restored ≈ input atol=1e-10
    end
finally
    MPI.Finalize()
end
