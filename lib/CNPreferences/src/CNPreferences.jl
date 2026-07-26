module CNPreferences
using Preferences
using LegatePreferences

const DEVEL_DEFAULT_WRAPPER_BRANCH = "main"

LegatePreferences.@make_preferences("cunumeric_")

# Compile-time: flipping it recompiles cuNumeric, so set it then load in a fresh process.
const FUSE_BROADCAST = @load_preference("FUSE_BROADCAST_EXPRS", true)
# Fuse when the broadcast tree has at least this many `Broadcasted` nodes (ops).
# Default 2 → single ops like `y .= cos.(x)` stay on the unfused path (no PTX compile).
# Set to 1 (preference or ENV) to fuse every eligible expression (e.g. in tests).
const FUSE_BROADCAST_MIN_OPS = let
    env = get(ENV, "CUNUMERIC_FUSE_BROADCAST_MIN_OPS", "")
    if !isempty(env)
        parse(Int, env)
    else
        @load_preference("FUSE_BROADCAST_MIN_OPS", 2)
    end
end
const TASK_SCOPE_NAMES = @load_preference("TASK_SCOPE_NAMES", false)

"""
    set_broadcast_fusion!(enabled::Bool; export_prefs=false, force=true)

Enable or disable broadcast fusion. When enabled (the default), eligible nested
broadcast expressions compile to a single CUDA PTX kernel.

Restart Julia after changing this preference.
"""
function set_broadcast_fusion!(enabled::Bool; export_prefs=false, force=true)
    return set_preferences!(@__MODULE__, "FUSE_BROADCAST_EXPRS" => enabled; export_prefs, force)
end

"""
    enable_broadcast_fusion!(; export_prefs=false, force=true)

Enable broadcast fusion. This is the default.
"""
enable_broadcast_fusion!(; kwargs...) = set_broadcast_fusion!(true; kwargs...)

"""
    disable_broadcast_fusion!(; export_prefs=false, force=true)

Disable broadcast fusion so each broadcast node runs as a separate cuNumeric op.
"""
disable_broadcast_fusion!(; kwargs...) = set_broadcast_fusion!(false; kwargs...)

"""
    set_broadcast_fusion_min_ops!(n::Integer; export_prefs=false, force=true)

Fuse only when a broadcast tree has at least `n` ops. Default is `2`, so single
ops such as `y .= cos.(x)` stay on the unfused path. Set `n = 1` to fuse those
too. Values must be `>= 1`.

Restart Julia after changing this preference. The environment variable
`CUNUMERIC_FUSE_BROADCAST_MIN_OPS` overrides this preference when set.
"""
function set_broadcast_fusion_min_ops!(n::Integer; export_prefs=false, force=true)
    n >= 1 || throw(ArgumentError("FUSE_BROADCAST_MIN_OPS must be >= 1, got $n"))
    return set_preferences!(
        @__MODULE__, "FUSE_BROADCAST_MIN_OPS" => Int(n); export_prefs, force
    )
end

"""
    set_task_scope_names!(enabled::Bool; export_prefs=false, force=true)

Enable or disable named Legate task scopes for debugging. Default is off.

When enabled, cuNumeric wraps ops in `Legate.with_scope` so provenance labels
appear in Legate logs/profiles. Pair with `LEGATE_CONFIG` flags such as
`--logging legate=debug --log-to-file` (set before Julia starts). Requires a
fresh Julia process after changing the preference.
"""
function set_task_scope_names!(enabled::Bool; export_prefs=false, force=true)
    return set_preferences!(@__MODULE__, "TASK_SCOPE_NAMES" => enabled; export_prefs, force)
end

"""
    enable_task_scope_names!(; export_prefs=false, force=true)

Enable named Legate task scopes for debugging. See [`set_task_scope_names!`](@ref).
"""
enable_task_scope_names!(; kwargs...) = set_task_scope_names!(true; kwargs...)

"""
    disable_task_scope_names!(; export_prefs=false, force=true)

Disable named Legate task scopes. This is the default.
"""
disable_task_scope_names!(; kwargs...) = set_task_scope_names!(false; kwargs...)

end # module CNPreferences
