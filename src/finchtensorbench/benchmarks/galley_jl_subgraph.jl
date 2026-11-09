# Worker for galley_jl_subgraph.py: times Galley.jl (Finch.jl backend) on a
# subgraph-matching workload, loaded with Galley.jl's own experiment code.
#
#   julia --project=<Galley.jl> galley_jl_subgraph.jl <workload> <first> <last> <runs>
#
# Run with Galley.jl as the working directory (its loaders use relative paths).
# Queries <first>..<last> (1-based, in Galley.jl's load order) are each run once
# to compile their kernels and then <runs> more times.  Lines starting with "@@"
# are the protocol the supervisor reads; everything else is ignored.

using SparseArrays
using Finch
using Galley
using Galley: FAQ_OPTIMIZERS, relabel_input, reindex_stats
using Statistics

include(joinpath(pwd(), "Experiments", "experiment_params.jl"))
include(joinpath(pwd(), "Experiments", "subgraph_workload.jl"))

function count_of(value)
    value isa Number && return value
    value isa AbstractVector && length(value) == 1 && return count_of(value[1])
    return count_of(value[])
end

function main(args)
    workload = getfield(Main, Symbol(args[1]))::WORKLOAD
    first_q, last_q, runs = parse.(Int, args[2:4])
    queries = load_subgraph_workload(workload, DCStats, nothing)
    println("@@READY\t", length(queries))
    flush(stdout)
    run_query(q) = galley(q.query, ST=DCStats; faq_optimizer=pruned, verbose=0)
    for i in first_q:min(last_q, length(queries))
        q = queries[i]
        println("@@BEGIN\t", i, "\t", q.query_path)
        flush(stdout)
        status, count, first_s, opt, exec, err = "ok", "", NaN, Float64[], Float64[], ""
        try
            first_s = @elapsed run_query(q)
            for _ in 1:runs
                result = run_query(q)
                push!(opt, result.opt_time)
                push!(exec, result.execute_time)
                count = string(count_of(result.value))
            end
        catch e
            status = e isa OutOfMemoryError ? "oom" : "error"
            err = replace(sprint(showerror, e), r"\s+" => " ")[1:min(end, 2000)]
        end
        expected = isnothing(q.expected_result) ? "" : string(q.expected_result)
        println(join(["@@RESULT", i, q.query_path, status, expected, count, first_s,
                      join(opt, ";"), join(exec, ";"), err], "\t"))
        flush(stdout)
    end
    println("@@DONE")
    flush(stdout)
end

main(ARGS)
