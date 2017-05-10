# This file is a part of Julia. License is MIT: https://julialang.org/license

script = """
# Issue #11948
f(x) = x+1
workspace()
@assert @__MODULE__() === Main
@assert isdefined(Main, :f)
@assert !@isdefined LastMain
@eval Core.Main begin
    @assert @__MODULE__() === Main
    @assert !isdefined(Main, :f)
    LastMain.f(2)

    # PR #12990
    io = IOBuffer()
    show(io, Pair)
    @assert String(take!(io)) == "Pair"
    @assert !Base.inbase(LastMain)
end
"""
exename = Base.julia_cmd()
run(`$exename --startup-file=no -e $script`, DevNull, STDOUT, STDERR)

# issue #17764
script2 = """
mutable struct Foo end
workspace()
@eval Core.Main begin
    mutable struct Foo end
    @assert Tuple{Type{LastMain.Foo}} !== Tuple{Type{Main.Foo}}
end
"""
run(`$exename --startup-file=no -e $script2`, DevNull, STDOUT, STDERR)
