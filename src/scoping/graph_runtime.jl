# Runtime guards shared by graph-generated optimized and fallback plans.
_fusion_disjoint(a, b) = isbitstype(typeof(a)) || isbitstype(typeof(b))
_fusion_disjoint(a::AbstractArray, b::AbstractArray) = !Base.mightalias(a, b)
_fusion_disjoint(a::NDArray, b::AbstractArray) = false
_fusion_disjoint(a::AbstractArray, b::NDArray) = false
_fusion_disjoint(a::NDArray, b::NDArray) = !nda_overlaps(a, b)

_all_ndarrays(xs...) = all(x -> x isa NDArray, xs)

# A copied read hoisted before an earlier write must come from disjoint storage.
function _hoisted_read_fresh(dest, root, read)
    return (read isa NDArray && _is_ndarray_slice(read)) || _fusion_disjoint(dest, root)
end
