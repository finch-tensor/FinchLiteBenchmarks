# Worker for galley_jl_subgraph.py: times Galley.jl on a subgraph-matching
# workload.  Galley.jl comes from Julia's package registry, pinned by the
# project in galley_jl/ next to this file:
#
#   julia --project=galley_jl galley_jl_subgraph.jl <manifest> <first> <last> <runs>
#
# The manifest, written by the supervisor, names the data graph and the queries:
#
#   graph<TAB><gcare|sgm><TAB><path>
#   query<TAB><path>            (one line per query, in order)
#
# Graphs and queries are loaded as Galley.jl's experiments load them
# (Experiments/subgraph_workload.jl): one 0/1 vector per vertex label, one 0/1
# matrix per edge label, and per query a sum over all query vertices of the
# product of its vertex-label vectors and edge-label matrices.  G-CARE graphs
# have directed, labelled edges and any number of labels per vertex; Sun & Luo
# ("sgm") graphs one label per vertex and undirected, unlabelled edges.
#
# Queries <first>..<last> (1-based) are each run once to compile their kernels
# and then <runs> more times.  Lines starting with "@@" are the protocol the
# supervisor reads; everything else is ignored.

using SparseArrays
using Galley
using Galley.Optimizer
using Galley.Optimizer: relabel_input

function load_graph(path, kind)
    sgm = kind == "sgm"
    n = 0
    edges = Dict{Tuple{Int,Int},Int}()
    vertices = Dict{Int,Vector{Int}}()
    edge_labels, vertex_labels = Set{Int}(), Set{Int}()
    for line in eachline(path)
        isempty(line) && continue
        parts = split(line)
        if line[1] == 'v'
            labels = sgm ? [parse(Int, parts[3])] : [parse(Int, x) for x in parts[3:end]]
            union!(vertex_labels, labels)
            vertices[parse(Int, parts[2]) + 1] = labels
            n += 1
        elseif line[1] == 'e'
            e1, e2 = parse(Int, parts[2]) + 1, parse(Int, parts[3]) + 1
            if sgm
                edges[(e1, e2)] = 0
                edges[(e2, e1)] = 0
                push!(edge_labels, 0)
            else
                label = parse(Int, parts[4])
                edges[(e1, e2)] = label
                push!(edge_labels, label)
            end
        end
    end
    vertex_vectors = Dict{Int,Any}()
    for label in vertex_labels
        ids = [v for (v, labels) in vertices if label in labels]
        vec = Tensor(SparseList(Element(0), n))
        copyto!(vec, sparsevec(ids, ones(Int, length(ids)), n))
        input = Input(vec, :i, "v_$label")
        input.stats = DCStats(input.tns.val, [:i])
        vertex_vectors[label] = input
    end
    edge_matrices = Dict{Int,Any}()
    for label in edge_labels
        pairs = [e for (e, l) in edges if l == label]
        mat = Tensor(Dense(SparseList(Element(0), n), n))
        copyto!(mat, sparse(first.(pairs), last.(pairs), ones(Int, length(pairs)), n, n))
        input = Input(mat, :i, :j, "e_$label")
        input.stats = DCStats(input.tns.val, [:i, :j])
        edge_matrices[label] = input
    end
    return vertex_vectors, edge_matrices
end

query_idx(i) = Symbol("v_", i)

# A query file: "v id label ..." and "e src dst label" lines (a label of -1
# matches anything, so adds no factor).
function load_query(path, vertex_vectors, edge_matrices)
    query_vertices = Dict{Int,Int}()
    query_edges = Dict{Tuple{Int,Int},Int}()
    for line in eachline(path)
        isempty(line) && continue
        parts = split(line)
        if line[1] == 'v'
            query_vertices[parse(Int, parts[2]) + 1] = parse(Int, parts[3])
        elseif line[1] == 'e'
            e1, e2 = parse(Int, parts[2]) + 1, parse(Int, parts[3]) + 1
            query_edges[(e1, e2)] = parse(Int, parts[4])
        end
    end
    factors = Set{PlanNode}()
    for (v, label) in query_vertices
        label == -1 && continue
        push!(factors, relabel_input(vertex_vectors[label], query_idx(v)))
    end
    for ((v1, v2), label) in query_edges
        label == -1 && continue
        push!(factors, relabel_input(edge_matrices[label], query_idx(v1), query_idx(v2)))
    end
    idxs = [query_idx(v) for v in keys(query_vertices)]
    return Query(:out, Materialize(Aggregate(+, 0, idxs..., MapJoin(*, factors...))))
end

function count_of(value)
    value isa Number && return value
    value isa AbstractVector && length(value) == 1 && return count_of(value[1])
    return count_of(value[])
end

function main(args)
    manifest = [split(line, '\t') for line in eachline(args[1]) if !isempty(line)]
    first_q, last_q, runs = parse.(Int, args[2:4])
    _, kind, graph_path = only(filter(l -> l[1] == "graph", manifest))
    query_paths = [String(l[2]) for l in manifest if l[1] == "query"]
    vertex_vectors, edge_matrices = load_graph(graph_path, kind)
    run_query(q) = galley(q; ST=DCStats, faq_optimizer=pruned, verbose=0)
    query(i) = load_query(query_paths[i], vertex_vectors, edge_matrices)
    # Warm up, as the finch worker does, so Julia start-up and Galley's generic
    # compilation are not charged to the first timed query: run one query
    # outside this worker's block (the workload's first, or its second when the
    # block starts there) and discard it.
    if length(query_paths) > 1
        try
            run_query(query(first_q == 1 ? 2 : 1))
        catch
        end
    end
    println("@@READY\t", length(query_paths))
    flush(stdout)
    for i in first_q:min(last_q, length(query_paths))
        println("@@BEGIN\t", i)
        flush(stdout)
        status, count, first_s, opt, exec, err = "ok", "", NaN, Float64[], Float64[], ""
        try
            q = query(i)
            first_s = @elapsed run_query(q)
            println("@@COLD\t", i)
            flush(stdout)
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
        println(join(["@@RESULT", i, status, count, first_s,
                      join(opt, ";"), join(exec, ";"), err], "\t"))
        flush(stdout)
    end
    println("@@DONE")
    flush(stdout)
end

main(ARGS)
