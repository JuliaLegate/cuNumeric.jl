using Libdl

# Load the C library
const lib_path = "./c/libarray_allocator.so"
const lib = Libdl.dlopen(lib_path)

# Get function pointers
const allocate_array_ptr = Libdl.dlsym(lib, :allocate_array)
const set_fptr = Libdl.dlsym(lib, :set)


function allocate_c_array(size::Int)
    ptr = ccall(allocate_array_ptr , Ptr{Cdouble}, (Cint,), size)
    arr = unsafe_wrap(Array, ptr, size, own=true)
    return arr
end


function set(arr::Array, index::Int, val::Float64)
    ccall(set_fptr, Ptr{Cvoid}, (Ptr{Cdouble}, Cint, Cdouble), arr, index, val)
end

# Test function to create arrays and let GC collect them
function test_gc(n_iter, size = 500_000)
    
    # Create several arrays
    for i in 1:n_iter
        arr = allocate_c_array(size)
        set(arr, 1, 1.0)
        GC.gc(false) 
    end
end

@time test_gc(1000)

println("Done")
Libdl.dlclose(lib)