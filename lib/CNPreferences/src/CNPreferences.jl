module CNPreferences
using Preferences
using LegatePreferences

const PREFS_CHANGED = Ref(false)
const DEPS_LOADED = Ref(false)

# default
const MODE_JLL = "jll"
# will compile wrappers from src
const MODE_DEVELOPER = "developer"
# not well tested, allows conda env install
const MODE_CONDA = "conda"

# Store what the values were when module loaded
const mode = @load_preference("mode")
# used for developer mode
const wrapper_branch = @load_preference("wrapper_branch")
const use_cupynumeric_jll = @load_preference("use_cupynumeric_jll")
const cupynumeric_path = @load_preference("cupynumeric_path")

# used for conda mode
const conda_env = @load_preference("conda_env")

# default developer options
const DEVEL_DEFAULT_JLL_WRAP_CONFIG = false
const DEVEL_DEFAULT_WRAPPER_BRANCH = "main"
const DEVEL_DEFAULT_JLL_CONFIG = true
const DEVEL_DEFAULT_CUPYNUMERIC_PATH = nothing

# from MPIPreferences.jl
"""
    CNPreferences.check_unchanged()

Throws an error if the preferences have been modified in the current Julia
session, or if they are modified after this function is called.

This is should be called from the `__init__()` function of any package which
relies on the values of CNPreferences.
"""
function check_unchanged()
    if PREFS_CHANGED[]
        error(
            "CNPreferences have changed, you will need to restart Julia for the changes to take effect"
        )
    end
    DEPS_LOADED[] = true
    return nothing
end

"""
    CNPreferences.use_conda(conda_env::String; export_prefs = false, force = true)

Tells cuNumeric.jl to use existing conda install. We make no gurantees of compiler compatability at this time.

Expects `conda_env` to be the absolute path to the root of the environment.
For example, `/home/julialegate/.conda/envs/cunumeric-gpu`
"""
function use_conda(conda_env::String; transitive=true, export_prefs=false, force=true)
    set_preferences!(CNPreferences,
        "conda_env" => conda_env,
        "mode" => MODE_CONDA;
        export_prefs=export_prefs,
        force=force,
    )
    if (transitive == true)
        # add transitive support to preferences
        LegatePreferences.use_conda(conda_env)
    end

    if conda_env == CNPreferences.conda_env && CNPreferences.mode == MODE_CONDA
        @info "CNPreferences found no differences."
    else
        PREFS_CHANGED[] = true
        @info "CNPreferences set to use local conda env at:" conda_env

        if DEPS_LOADED[]
            error("You will need to restart Julia for the changes to take effect.")
        end
    end
end

"""
    CNPreferences.use_jll_binary(; export_prefs = false, force = true)

Tells Legate.jl to use JLLs. This is the default option. 
"""
function use_jll_binary(; transitive=true, export_prefs=false, force=true)
    set_preferences!(CNPreferences,
        "mode" => MODE_JLL;
        export_prefs=export_prefs,
        force=force,
    )
    if (transitive == true)
        # add transitive support to preferences
        LegatePreferences.use_jll_binary(; export_prefs=export_prefs, force=force)
    end
    if CNPreferences.mode == MODE_JLL
        @info "CNPreferences found no differences. Using JLLs."
    else
        PREFS_CHANGED[] = true
        @info "CNPreferences set to use JLLs."

        if DEPS_LOADED[]
            error(
                "You will need to restart Julia for the changes to take effect. JLLs do not require building."
            )
        end
    end
end

"""
    CNPreferences.use_developer_mode(; wrapper_branch="main", use_cupynumeric_jll=true, cupynumeric_path=nothing, export_prefs = false, force = true)

Tells cuNumeric.jl to enable developer mode. This will clone cunumeric_jl_wrapper into cuNumeric.jl/deps. 

To specify a cunumeric_jl_wrapper branch: ```wrapper_branch="some-branch"```
To disable using cupynumeric_jll: ```use_cupynumeric_jll=false``` 
If you disable cupynumeric_jll, then you need to set a path to cuPyNumeric with ```cupynumeric_path="/path/to/cupynumeric"```
"""
function use_developer_mode(;
    wrapper_branch=DEVEL_DEFAULT_WRAPPER_BRANCH,
    use_cupynumeric_jll=DEVEL_DEFAULT_JLL_CONFIG,
    cupynumeric_path=DEVEL_DEFAULT_CUPYNUMERIC_PATH,
    export_prefs=false,
    force=true,
)
    if (use_cupynumeric_jll == false)
        if (cupynumeric_path == nothing)
            error("You must set a cupynumeric_path if you are disabling use_cupynumeric_jll")
        end
    end

    set_preferences!(CNPreferences,
        "mode" => MODE_DEVELOPER,
        "wrapper_branch" => wrapper_branch,
        "use_cupynumeric_jll" => use_cupynumeric_jll,
        "cupynumeric_path" => cupynumeric_path;
        export_prefs=export_prefs,
        force=force,
    )

    # add transitive support to preferences
    @warn "
        CNPreferences Developer mode does not support transitive developer mode for Legate.
        To configure Legate for developer mode, you need to use LegatePreferences.use_developer_mode()
    "

    same_branch = wrapper_branch == CNPreferences.wrapper_branch
    same_jll_conf = use_cupynumeric_jll == CNPreferences.use_cupynumeric_jll
    same_cupynumeric_path = cupynumeric_path == CNPreferences.cupynumeric_path
    same_mode = CNPreferences.mode == MODE_DEVELOPER

    if same_branch && same_jll_conf && same_cupynumeric_path && same_mode
        @info "CNPreferences found no differences. Using Developer mode."
    else
        PREFS_CHANGED[] = true
        @info "CNPreferences set to use developer mode.."

        if DEPS_LOADED[]
            error(
                "You will need to restart Julia for the changes to take effect. You need to call Pkg.build()"
            )
        end
    end
end

end # module CNPreferences
