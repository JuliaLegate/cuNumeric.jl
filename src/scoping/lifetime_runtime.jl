"""
    _lifetime_dependencies(value) -> Tuple

Internal extension point for `@accelerate`. Return the values whose handles
`value` borrows. Dependencies must form an acyclic graph and stay unchanged
while a binding is live. Declaring a dependency does not transfer ownership.

An NDArray owns its own backend handle, even when its store is a view, so it
has no borrowed handle dependencies. Julia wrappers specialize this method.
"""
_lifetime_dependencies(::Any) = ()
_lifetime_dependencies(x::Tuple) = x
_lifetime_dependencies(x::NamedTuple) = values(x)

_lifetime_handles(x::NDArray) = (x,)
function _lifetime_handles(x)
    return mapfoldl(_lifetime_handles, (a, b) -> (a..., b...),
                   _lifetime_dependencies(x); init=())
end

mutable struct _LifetimeHandle
    uses::Int
    owned::Bool
    protected::Bool
end

struct _LifetimeScope
    bindings::Dict{Symbol,Tuple}
    handles::IdDict{NDArray,_LifetimeHandle}
end
_LifetimeScope() = _LifetimeScope(Dict{Symbol,Tuple}(), IdDict{NDArray,_LifetimeHandle}())

function _lifetime_handle!(scope::_LifetimeScope, handle::NDArray)
    return get!(scope.handles, handle) do
        _LifetimeHandle(0, false, false)
    end
end

# Protect caller-owned handles and dependencies of values escaping the scope.
function _lifetime_protect!(scope::_LifetimeScope, value)
    for handle in _lifetime_handles(value)
        _lifetime_handle!(scope, handle).protected = true
    end
    return value
end

function _lifetime_release!(scope::_LifetimeScope, handles::Tuple)
    for handle in handles
        entry = scope.handles[handle]
        entry.uses -= 1
        if entry.uses == 0 && !entry.protected
            entry.owned && destroy!(handle)
            delete!(scope.handles, handle)
        end
    end
    return nothing
end

# Snapshot dependencies per binding: aliases count independently, and rebinding
# one name cannot release a handle still needed by another. Acquire the new
# references before releasing the old ones, including for `x = wrapper(x)`
# and swaps/destructuring.
function _lifetime_bind!(scope::_LifetimeScope, names::Tuple, values::Tuple, protected::Tuple)
    previous = map(name -> get(scope.bindings, name, ()), names)
    dependencies = map(_lifetime_handles, values)
    for (name, value, handles, protect) in zip(names, values, dependencies, protected)
        for handle in handles
            entry = _lifetime_handle!(scope, handle)
            entry.uses += 1
            entry.owned |= handle === value
            entry.protected |= protect
        end
        scope.bindings[name] = handles
    end
    foreach(handles -> _lifetime_release!(scope, handles), previous)
    return nothing
end

function maybe_insert_delete(scope::_LifetimeScope, name::Symbol)
    _lifetime_release!(scope, pop!(scope.bindings, name, ()))
    return nothing
end

function _lifetime_finish!(scope::_LifetimeScope)
    for (handle, entry) in scope.handles
        entry.owned && !entry.protected && destroy!(handle)
    end
    empty!(scope.bindings)
    empty!(scope.handles)
    return nothing
end
