using Test

using HeFFTe_jll
using Heffte: Box3D, count, proc_setup_min_surface, split_world

@testset "Heffte Tests" begin
    @testset "Box3D" begin
        @test Box3D((1, 2, 3)).size == (1, 2, 3)
        @test count(Box3D((1, 2, 3))) == 6
    end

    @testset "proc_setup_min_surface" begin
        @testset "Example usage" begin
            world_box = Box3D((100, 100, 10))
            num_procs = 16
            grid = proc_setup_min_surface(world_box, num_procs)
            @test grid == (4, 4, 1)
        end

        @testset "Edge cases" begin
            world_box = Box3D((1, 1, 1))
            num_procs = 1
            grid = proc_setup_min_surface(world_box, num_procs)
            @test grid == (1, 1, 1)

            world_box = Box3D((2, 1, 1))
            num_procs = 2
            grid = proc_setup_min_surface(world_box, num_procs)
            @test grid == (2, 1, 1)

            world_box = Box3D((1, 2, 1))
            num_procs = 2
            grid = proc_setup_min_surface(world_box, num_procs)
            @test grid == (1, 2, 1)

            world_box = Box3D((1, 1, 2))
            num_procs = 2
            grid = proc_setup_min_surface(world_box, num_procs)
            @test grid == (1, 1, 2)

            world_box = Box3D((2, 1, 2))
            num_procs = 4
            grid = proc_setup_min_surface(world_box, num_procs)
            @test grid == (2, 1, 2)

            world_box = Box3D((2, 1, 1))
            num_procs = 4
            @test_throws AssertionError proc_setup_min_surface(world_box, num_procs)
        end
    end

    @testset "HeFFTe_jll" begin
        if Sys.islinux() && Sys.ARCH in (:x86_64, :aarch64)
            @test isdefined(HeFFTe_jll, :libheffte)
        else
            @test_skip isdefined(HeFFTe_jll, :libheffte)
        end
    end

    @testset "split_world" begin
        world_box = Box3D((1, 1, 1), (4, 4, 4))
        proc_grid = (2, 2, 2)
        subdomains = split_world(world_box, proc_grid)
        @test length(subdomains) == 8
        @test subdomains[1].low == (1, 1, 1)
        @test subdomains[1].high == (2, 2, 2)
        @test subdomains[8].low == (3, 3, 3)
        @test subdomains[8].high == (4, 4, 4)
    end
end
