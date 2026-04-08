export get_time_microseconds, get_time_nanoseconds

@doc"""
Returns the timestamp in microseconds. Blocks on all Legate operations
preceding the call to this function.
"""
function get_time_microseconds()
    return Legate.time_microseconds()
end

@doc"""
Returns the timestamp in nanoseconds. Blocks on all Legate operations
preceding the call to this function.
"""
function get_time_nanoseconds()
    return Legate.time_nanoseconds()
end

function Experimental(setting::Bool)
    task_local_storage(:Experimental, setting)
end

function assert_experimental()
    if get(task_local_storage(), :Experimental, false) !== true
        throw(
            ArgumentError(
                "Experimental features are disabled." *
                " Use `cuNumeric.Experimental(true)` to enable them.",
            ),
        )
    end
end
