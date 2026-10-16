# Julia 1.10's `Broadcast.flatten` nests varargs closures that inference gives up
# on for deep trees, leaving dynamic calls GPUCompiler rejects. Backport the
# inferable `Pick`-based implementation from Julia 1.11.
@static if VERSION >= v"1.11"
    const _broadcast_flatten = Base.Broadcast.flatten
else
    function _broadcast_flatten(bc::Base.Broadcast.Broadcasted)
        Base.Broadcast.isflat(bc) && return bc
        args = Base.Broadcast.cat_nested(bc)
        makeargs = _make_makeargs(bc.args, 1)[1]
        f = Base.maybeconstructor(bc.f)
        newf = (args...) -> (@inline; f(_prepare_args(makeargs, args)...))
        return Base.Broadcast.Broadcasted(bc.style, newf, args, bc.axes)
    end

    @inline function _make_makeargs(args::Tuple, n::Int)
        head, n = _make_makeargs1(args[1], n)
        rest, n = _make_makeargs(Base.tail(args), n)
        return (head, rest...), n
    end
    _make_makeargs(::Tuple{}, n::Int) = (), n

    struct _Pick{N} <: Function end
    (::_Pick{N})(@nospecialize(args::Tuple)) where {N} = args[N]

    @inline _make_makeargs1(_, n::Int) = _Pick{n}(), n + 1
    @inline function _make_makeargs1(bc::Base.Broadcast.Broadcasted, n::Int)
        makeargs, n = _make_makeargs(bc.args, n)
        f = Base.maybeconstructor(bc.f)
        makeargs1 = (args::Tuple) -> (@inline; f(_prepare_args(makeargs, args)...))
        return makeargs1, n
    end

    @inline _prepare_args(makeargs::Tuple, @nospecialize(x::Tuple)) =
        (makeargs[1](x), _prepare_args(Base.tail(makeargs), x)...)
    @inline _prepare_args(makeargs::Tuple{Any}, @nospecialize(x::Tuple)) = (makeargs[1](x),)
    _prepare_args(::Tuple{}, ::Tuple) = ()
end
