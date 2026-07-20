module CNPreferences
using Preferences
using LegatePreferences

const DEVEL_DEFAULT_WRAPPER_BRANCH = "main"

LegatePreferences.@make_preferences("cunumeric_")

# Compile-time: flipping it recompiles cuNumeric, so set it then load in a fresh process.
const FUSE_BROADCAST = @load_preference("FUSE_BROADCAST_EXPRS", true)
const TASK_SCOPE_NAMES = @load_preference("TASK_SCOPE_NAMES", false)

function set_broadcast_fusion!(enabled::Bool; export_prefs=false, force=true)
    return set_preferences!(@__MODULE__, "FUSE_BROADCAST_EXPRS" => enabled; export_prefs, force)
end
enable_broadcast_fusion!(; kwargs...) = set_broadcast_fusion!(true; kwargs...)
disable_broadcast_fusion!(; kwargs...) = set_broadcast_fusion!(false; kwargs...)

function set_task_scope_names!(enabled::Bool; export_prefs=false, force=true)
    return set_preferences!(@__MODULE__, "TASK_SCOPE_NAMES" => enabled; export_prefs, force)
end
enable_task_scope_names!(; kwargs...) = set_task_scope_names!(true; kwargs...)
disable_task_scope_names!(; kwargs...) = set_task_scope_names!(false; kwargs...)

end # module CNPreferences
