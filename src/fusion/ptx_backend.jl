# const PTX_FUSION_CACHE = Dict{UInt64, CUDATask}()
# const PTX_FUSION_KWARGS = (:blocks, :threads)

# macro ptx_debug(ex)
#     # PRINT GENERATED PTX TO STDOUT
# end

# function _fuse_broadcast(ex::Expr, ::Type{PTXBackend})

#     call = ex[end]
#     kwargs = map(ex[1:end-1]) do kwarg
#         if kwarg in FUSE_KWARGS
#             :($kwarg = $kwarg)
#         else
#             throw(ArgumentError("Invalid keyword argument '$kwarg', expected one of $(FUSE_KWARGS)"))
#         end
#     end

#     #! TODO SET DEFAULT BLOCKS/THREADS
#     #! HOW TO KNOW WHATS THE RIGHT DIMENSION??

#     blocks = get(kwargs[:blocks], DEFAULT_BLOCKS)
#     threads = get(kwargs[:threads], DEFAULT_THREADS)

#     # I think its safe to not put kwargs in the hash
#     # as the PTX does not specialize on blocks/threads
#     #! NEED TO PUT TYPE INFO OF INPUTS/OUTPUTS HERE
#     cache_key = hash(call)

#     return esc(quote

#         local task::CUDATask
#         if haskey(PTX_FUSION_CACHE, cache_key)
#             task = PTX_FUSION_CACHE[cache_key]
#         else
#             local _buf = IOBuffer()
#             # generate ptx using CUDA.jl
#             CUDATools.@device_code_ptx io=_buf $(call)

#             local _ptx = String(take!(_buf))
#             local _func_name = extract_kernel_name(_ptx)

#             # issue ptx_task within legate runtime to register cufunction ptr with cucontext
#             ptx_task(_ptx, _func_name)

#             PTX_FUSION_CACHE[cache_key] = CUDATask(_func_name, _types)
#         end
#     end)

# end

# function CUDA.code_ptx(
#         io::IO,
#         @nospecialize(func),
#         @nospecialize(types);
#         kernel=false,
#         kwargs...
#     )
#     compiler_kwargs, kwargs = split_kwargs_runtime(kwargs, CUDACore.COMPILER_KWARGS)
#     source = methodinstance(typeof(func), Base.to_tuple_type(types))
#     config = CUDACore.compiler_config(device(); kernel, compiler_kwargs...)
#     job = CompilerJob(source, config)
#     # use frozen world to avoid recompiling the compiler infrastructure
#     CUDACore.invoke_frozen(code_ptx, $(args...); kwargs...)
# end
