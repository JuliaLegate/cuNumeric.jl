function assert_active_model(expected::Symbol)
    actual = get(ENV, "CUNUMERIC_BENCH_ACTIVE_MODEL", "")
    return actual == string(expected) || error(
        "The $(expected) worker must be launched by run_benchmark.sh with " *
        "--model=$(expected); active model is $(repr(actual))",
    )
end

function assert_models_not_loaded(forbidden::Tuple)
    loaded = Set(id.name for id in keys(Base.loaded_modules))
    conflicts = sort!(collect(intersect(loaded, Set(forbidden))))
    return isempty(conflicts) || error(
        "Execution-model isolation violated; conflicting packages loaded: " * join(conflicts, ", ")
    )
end
