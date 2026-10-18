# Base's `flatten` nests closures that 1.10 can't infer on deep trees, leaving
# dynamic calls GPUCompiler rejects. Evaluate the tree by plain recursion instead.
@static if VERSION >= v"1.11"
    const _broadcast_flatten = Base.Broadcast.flatten
else
    struct _ArgRef{N} end

    struct _CallNode{F,Args}
        f::F
        args::Args
    end

    struct _TreeCall{T} <: Function
        tree::T
    end

    @inline _eval_tree(::_ArgRef{N}, args) where {N} = args[N]
    @inline _eval_tree(node::_CallNode, args) = node.f(_eval_children(node.args, args)...)

    @inline _eval_children(::Tuple{}, args) = ()
    @inline _eval_children(nodes::Tuple, args) =
        (_eval_tree(first(nodes), args), _eval_children(Base.tail(nodes), args)...)

    (t::_TreeCall)(args...) = _eval_tree(t.tree, args)

    _build_children(::Tuple{}, n) = (), n
    function _build_children(args::Tuple, n)
        head, n = _build_tree(first(args), n)
        rest, n = _build_children(Base.tail(args), n)
        return (head, rest...), n
    end

    # Leaves are numbered left to right, matching `cat_nested`.
    _build_tree(_, n) = _ArgRef{n}(), n + 1
    function _build_tree(bc::Base.Broadcast.Broadcasted, n)
        args, n = _build_children(bc.args, n)
        return _CallNode(Base.maybeconstructor(bc.f), args), n
    end

    function _broadcast_flatten(bc::Base.Broadcast.Broadcasted)
        Base.Broadcast.isflat(bc) && return bc
        tree, _ = _build_tree(bc, 1)
        return Base.Broadcast.Broadcasted(
            bc.style, _TreeCall(tree), Base.Broadcast.cat_nested(bc), bc.axes
        )
    end
end
