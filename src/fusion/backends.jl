abstract type FusionBackend end

"""
    PTXBackend

The default backend for broadcasted expressions. This backend
    leverages CUDA.jl to generate PTX code which is executed by the Legate runtime.
"""
struct PTXBackend <: FusionBackend end

# Backend for expressions that return `Broadcasted` objects.
# For example, `@fuse z .= x .+ y` uses this backend.
const DEFAULT_BROADCASTED_FUSION_BACKEND = PTXBackend
const SUPPORTED_BROADCASTED_FUSION_BACKENDS = (PTXBackend,)

#! Call this on init or in global scope
function set_broadcasted_fusion_backend()
    backend = load_preference(
        CNPreferences, "BCAST_FUSION_BACKEND", DEFAULT_BROADCASTED_FUSION_BACKEND
    )
    if backend ∉ SUPPORTED_BROADCASTED_FUSION_BACKENDS
        throw(
            ArgumentError(
                "Unsupported broadcasted fusion backend: $backend. Supported backends are: $(SUPPORTED_BROADCASTED_FUSION_BACKENDS)"
            ),
        )
    end
    return backend
end
