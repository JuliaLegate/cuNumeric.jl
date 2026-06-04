module CNPreferences
using Preferences
using LegatePreferences

const DEVEL_DEFAULT_WRAPPER_BRANCH = "main"

LegatePreferences.@make_preferences("cunumeric_")

# Compile-time: flipping it recompiles cuNumeric, so set it then load in a fresh process.
const FUSE_BROADCAST = @load_preference("FUSE_BROADCAST_EXPRS", true)

function set_broadcast_fusion!(enabled::Bool; export_prefs=false, force=true)
    set_preferences!(@__MODULE__, "FUSE_BROADCAST_EXPRS" => enabled; export_prefs, force)
end
enable_broadcast_fusion!(; kwargs...) = set_broadcast_fusion!(true; kwargs...)
disable_broadcast_fusion!(; kwargs...) = set_broadcast_fusion!(false; kwargs...)

end # module CNPreferences
