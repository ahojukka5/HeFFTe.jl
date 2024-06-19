using Heffte.API
using MPI

MPI.Init()
comm = MPI.COMM_WORLD

me = MPI.Comm_rank(comm)
num_ranks = MPI.Comm_size(comm)
println("Hello world, I am $me of $num_ranks")

if num_ranks != 2
    println("This example requires exactly 2 ranks")
    MPI.Finalize()
    exit(1)
end

inbox_low = [0, 0, 0]
inbox_high = [3, 3, 3]
inbox_order = [0, 1, 2]
outbox_low = [0, 0, 0]
outbox_high = [3, 3, 3]
outbox_order = [0, 1, 2]

if me == 0
    inbox_high[3] = 1
    outbox_high[3] = 1
else
    inbox_low[3] = 2
    outbox_low[3] = 2
end

backend = Heffte_BACKEND_FFTW
options = HefftePlanOptions(0, 0, 0, 0)
heffte_set_default_options(backend, options)
@show options

# Create a plan
plan_ref = Ref{Ptr{HefftePlan}}(C_NULL)
@show plan_ref

use_reorder = 1
use_pencils = 1
use_gpu_aware = 1
options = HefftePlanOptions(use_reorder, Heffte_RESHAPE_ALGORITHM_ALLTOALLV, use_pencils, 1)

status = heffte_plan_create(backend, inbox_low, inbox_high, inbox_order,
                            outbox_low, outbox_high, outbox_order,
                            comm, options, plan_ref)

#=
status = heffte_plan_create_r2c(backend, inbox_low, inbox_high, inbox_order,
                                outbox_low, outbox_high, outbox_order,
                                1, comm, options, plan_ref)
=#

plan_ptr = plan_ref[]
plan = unsafe_load(plan_ptr)
@show plan
@show plan_ptr
@show plan_ref

if status != Heffte_SUCCESS
    println("Failed at heffte_plan_create() with error code: ", status)
    MPI.Finalize()
    exit(1)
end

if me == 0
    println("Plan created successfully")
end

size_inbox = heffte_size_inbox(plan)
size_outbox = heffte_size_outbox(plan)
size_workspace = heffte_size_workspace(plan)

if me == 0
    if heffte_is_r2c(plan) == 1
        println("R2C plan")
    else
        println("C2C plan")
    end
    println("Size of inbox: ", size_inbox)
    println("Size of outbox: ", size_outbox)
    println("Size of workspace: ", size_workspace)
end

input = zeros(size_inbox)
for i=1:size_inbox
    input[i] = me*32 + i - 1
end

output = zeros(Complex{Float64}, size_outbox)

heffte_forward_d2z(plan, input, output, Heffte_SCALE_SYMMETRIC)

@info "Forward FFT result: $output"

input2 = zeros(size_inbox)

heffte_backward_z2d(plan, output, input2, Heffte_SCALE_SYMMETRIC)

@info "Backward FFT result: $input2"

@info "Original input: $input"

MPI.Finalize()
