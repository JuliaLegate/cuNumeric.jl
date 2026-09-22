using cuNumeric, LinearAlgebra
host = reshape(Float32.(1:1536*1024),1536,1024)
xh = ones(Float32,1024)
x = NDArray(xh)
y = cuNumeric.zeros(Float32,1536)
for layout in (:row,:column,:transposed_storage)
    if layout === :column
        store = cuNumeric.Legate.attach_external_col_major(host)
        ptr = cuNumeric.nda_store_to_ndarray(store.handle)
        finalize(store.handle)
        A = NDArray(ptr,Float32,Val(2),host)
    elseif layout === :transposed_storage
        A = permutedims(NDArray(permutedims(host)))
    else
        A = NDArray(host)
    end
    println("LAYOUT ",layout); flush(stdout)
    mul!(y,A,x)
    cuNumeric.issue_execution_fence(;block=true)
    @assert Array(y) ≈ host*xh
    println("VERIFIED ",layout); flush(stdout)
end
