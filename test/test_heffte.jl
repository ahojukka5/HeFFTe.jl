# test 'high level API'

using Heffte
using MPI
comm = MPI.COMM_WORLD

MPI.Init()

function meprintln(args...)
    if MPI.Comm_rank(comm) == 0
        println(args...)
    end
end

me = MPI.Comm_rank(comm)
num_ranks = MPI.Comm_size(comm)
meprintln("Hello world, I am $me of $num_ranks")

world = Heffte.Box3D((64, 64, 64))
count = Heffte.count(world)
meprintln("World has $count indices")

grid = Heffte.proc_setup_min_surface(world, num_ranks)
meprintln("Optimal grid dimensions: $grid")

boxes = Heffte.split_world(world, grid)
for (i, box) in enumerate(boxes)
    meprintln("Rank $i has box $box")
end

my_inbox = boxes[me+1]
my_outbox = boxes[me+1]

options = Heffte.PlanOptions(:FFTW)
meprintln("Options: $options")

plan = Heffte.Plan(my_inbox, my_outbox, options)
meprintln("Plan: $plan")

input = rand(Float64, Heffte.size_inbox(plan))
output = zeros(Complex{Float64}, Heffte.size_outbox(plan))
wrk = zeros(Complex{Float64}, Heffte.size_workspace(plan))

meprintln("Size of inbox: ", Heffte.size_inbox(plan))
meprintln("Size of outbox: ", Heffte.size_outbox(plan))
meprintln("Size of workspace: ", Heffte.size_workspace(plan))

Heffte.forward!(plan, input, output, wrk)
input2 = zeros(Float64, Heffte.size_inbox(plan))
Heffte.backward!(plan, output, input2, wrk)

using LinearAlgebra

meprintln("Error: ", norm(input - input2))

MPI.Finalize()
