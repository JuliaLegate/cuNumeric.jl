module cuNumericStructArraysExt

using cuNumeric
using StructArrays

const Broadcasted = Base.Broadcast.Broadcasted
const Extruded = Base.Broadcast.Extruded

# Project a struct-valued scalar function directly to one field. This keeps the
# temporary struct inside the GPU kernel instead of allocating an NDArray of it.
struct FieldFunction{field,F}
    f::F
end
@inline (p::FieldFunction{field})(args...) where {field} = getfield(p.f(args...), field)

_uses_component(x, components) = any(c -> x === c, components)
_uses_component(x::Extruded, components) = _uses_component(x.x, components)
_uses_component(x::Broadcasted, components) =
    any(arg -> _uses_component(arg, components), x.args)

function Base.copyto!(
    dest::StructArray{T}, bc::Broadcasted{<:cuNumeric.NDArrayStyle}
) where {T}
    components = Tuple(StructArrays.components(dest))
    all(c -> c isa cuNumeric.NDArray, components) ||
        throw(ArgumentError("StructArray broadcast requires NDArray field storage"))
    axes(dest) == axes(bc) || Base.Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest

    # A field may read another field of dest. Stage results before replacing any
    # component so in-place broadcasts retain their usual simultaneous semantics.
    aliases_dest = _uses_component(bc, components)
    outputs = aliases_dest ? map(similar, components) : components
    try
        for (name, out) in zip(fieldnames(T), outputs)
            projected = Broadcasted{cuNumeric.NDArrayStyle{ndims(dest)}}(
                FieldFunction{name,typeof(bc.f)}(bc.f), bc.args, bc.axes
            )
            cuNumeric.can_fuse_linear_broadcast(out, projected) || throw(
                ArgumentError("StructArray assignment requires same-shaped NDArray inputs")
            )
            cuNumeric.fuse_broadcast_tree!(out, projected)
        end
        if aliases_dest
            for (component, output) in zip(components, outputs)
                copyto!(component, output)
            end
        end
    finally
        aliases_dest && foreach(cuNumeric.destroy!, outputs)
    end
    return dest
end

end
