#----------------------------------------------------------
module Node2Vec
    using LightGraphs;
    using SimpleWeightedGraphs;
    using Random;
    using Word2Vec;
    using DelimitedFiles;
    import StatsBase: countmap;

    export simulate_walks;
    export learn_embeddings;

    #------------------------------------------------------
    struct alias
        X::Vector{Any};
        U::Vector{Float64};
        K::Vector{Int64};
    end
    #------------------------------------------------------

    #------------------------------------------------------
    function get_alias(X, probs)
        n = length(X);
        U = zeros(n);
        K = zeros(Int64,n);

        smalls = Vector{Int64}();
        larges = Vector{Int64}();

        for (i,prob) in enumerate(probs)
            U[i] = n*prob;

            if U[i] < 1.0
                push!(smalls,i);
            else
                push!(larges,i);
            end
        end

        while length(smalls) > 0 && length(larges) > 0
            small = pop!(smalls);
            large = pop!(larges);

            K[small]=large;
            U[large]=U[large]-(1.0-U[small]);
            if U[large] < 1.0
                push!(smalls,large);
            else
                push!(larges,large);
            end
        end

        return alias(collect(X), U, K);
    end
    #------------------------------------------------------

    #------------------------------------------------------
    function sample_alias(als::alias)
        n = length(als.X);
        i = Int(ceil(rand()*n));
        if rand() < als.U[i]
            return als.X[i];
        else
            return als.X[als.K[i]];
        end
    end
    #------------------------------------------------------

    #------------------------------------------------------
    function verify_alias_sampling(n)
        X = 1:n;
        probs = rand(length(X)); probs = probs ./ sum(probs);
        als = get_alias(X, probs);

        n = 1000000;

        samples = [];
        for _ in 1:n
            push!(samples, sample_alias(als));
        end
        sample_ct = countmap(samples);


        for i in 1:length(X)
            @assert abs(sample_ct[X[i]]/n - probs[i]) < 1.0e-2;
        end
    end
    #------------------------------------------------------

    #------------------------------------------------------
    function precompute_alias(g, p, q)

        function edge_alias(g, u, v, p, q)
            nbrs = neighbors(g,v);
            upbs = Vector{Float64}();

            for w in nbrs
                if w == u
                    push!(upbs, weights(g)[v,w]/p);
                elseif has_edge(g,w,u)
                    push!(upbs, weights(g)[v,w]);
                else
                    push!(upbs, weights(g)[v,w]/q);
                end
            end

            return get_alias(collect(nbrs), upbs./sum(upbs));
        end

        valias = Dict{Int64,alias}();
        for u in vertices(g)
            nbrs = neighbors(g,u);
            upbs = [weights(g)[u,v] for v in neighbors(g,u)];
            valias[u] = get_alias(nbrs, upbs./sum(upbs));
        end

        ealias = Dict{Tuple{Int64,Int64},alias}();
        if is_directed(g)
            for e in edges(g)
                ealias[(e.src,e.dst)] = edge_alias(g, e.src, e.dst, p, q);
            end
        else
            for e in edges(g)
                ealias[(e.src,e.dst)] = edge_alias(g, e.src, e.dst, p, q);
                ealias[(e.dst,e.src)] = edge_alias(g, e.dst, e.src, p, q);
            end
        end

        return valias, ealias;
    end
    #------------------------------------------------------

    #------------------------------------------------------
    function node2vec_walk(g, u, len, valias, ealias)
        walk = [u];
        while length(walk) < len
            if length(walk) == 1
                alias = valias[walk[end]];
            else
                alias = ealias[(walk[end-1],walk[end])];
            end

            if length(alias.X) != 0
                push!(walk, sample_alias(alias));
            else
                break;
            end
        end

        return walk;
    end
    #------------------------------------------------------

    #------------------------------------------------------
    function simulate_walks(g, num_rounds, len, p, q)
        valias, ealias = precompute_alias(g, p, q);
        walks = Vector{Vector{Int64}}();
        for _ in 1:num_rounds
            for u in vertices(g)
                push!(walks, node2vec_walk(g, u, len, valias, ealias))
            end
        end

        return walks;
    end
    #------------------------------------------------------

    #------------------------------------------------------
    function learn_embeddings(walks, ndim)
        str_walks=map(x->string.(x), walks);
        writedlm("/tmp/walks.txt", str_walks);
        word2vec("/tmp/walks.txt", "/tmp/vec.txt", size=ndim, verbose=true);
        model=wordvectors("/tmp/vec.txt");
        println();

        return model;
    end
    #------------------------------------------------------
end
#----------------------------------------------------------
