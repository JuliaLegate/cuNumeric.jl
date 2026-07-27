export get_time_microseconds, get_time_nanoseconds
export issue_mapping_fence, issue_execution_fence

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

"""
    issue_mapping_fence()

Insert a Legate mapping fence (DAG-only; does not block the Julia caller).
"""
issue_mapping_fence() = Legate.issue_mapping_fence()

"""
    issue_execution_fence(block::Bool)

Insert a Legate execution fence. `block=true` waits until prior ops finish;
`block=false` only inserts a DAG node (Julia can keep submitting).
"""
issue_execution_fence(; block::Bool=false) = Legate.issue_execution_fence(block)

function Experimental(setting::Bool)
    return task_local_storage(:Experimental, setting)
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
