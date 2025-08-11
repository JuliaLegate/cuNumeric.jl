export get_time_microseconds, get_time_nanoseconds

@doc"""
Returns the timestamp in microseconds. Blocks on all Legate operations
preceding the call to this function.
"""
function get_time_microseconds()
    return Legate.value(Legate.time_microseconds())
end

@doc"""
Returns the timestamp in nanoseconds. Blocks on all Legate operations
preceding the call to this function.
"""
function get_time_nanoseconds()
    return Legate.value(Legate.time_nanoseconds())
end
