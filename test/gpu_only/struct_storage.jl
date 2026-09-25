struct StructStorageTriple{T}
    a::T
    b::T
    c::T
end

struct StructStorageParticle{T}
    position::T
    velocity::T
end

_struct_storage_step(p, dt) = StructStorageParticle(
    p.position + dt * p.velocity, p.velocity
)

function _struct_storage_float(x)
    return StructStorageTriple{Float32}(
        Float32(x + 1), Float32(x + 2), Float32(x + 3)
    )
end
_struct_storage_uint(x) = StructStorageTriple{UInt64}(UInt64(x + 1), UInt64(x + 2), UInt64(x + 3))

struct StructStoragePacked
    a::UInt8
    b::Bool
    c::Int16
    d::UInt32
end
_struct_storage_packed(x) = StructStoragePacked(UInt8(x + 1), isodd(x), Int16(x + 3), UInt32(x + 4))

struct StructStoragePadded
    a::UInt8
    b::Float64
    c::Int16
    d::Float32
end
function _struct_storage_padded(x)
    return StructStoragePadded(
        UInt8(x + 1), Float64(x + 2), Int16(x + 3), Float32(x + 4)
    )
end

struct StructStorageComplex
    a::ComplexF32
    b::Float64
    c::UInt8
end
function _struct_storage_complex(x)
    return StructStorageComplex(
        ComplexF32(Float32(x + 1), Float32(x + 2)), Float64(x + 3), UInt8(x + 4)
    )
end
_struct_storage_real(x::StructStorageComplex) = real(x.a)
_struct_storage_imag(x::StructStorageComplex) = imag(x.a)

_struct_storage_named(x) = (a=Float32(x + 1), b=UInt8(x + 2))

struct StructStorageNested
    a::StructStorageTriple{Float32}
    b::Float64
end
_struct_storage_nested(x) = StructStorageNested(_struct_storage_float(x), Float64(x + 4))

struct StructStorageTupleField
    a::NTuple{3,Float32}
    b::Int32
end
function _struct_storage_tuple(x)
    return StructStorageTupleField(
        (Float32(x + 1), Float32(x + 2), Float32(x + 3)), Int32(x + 4)
    )
end

@inline _struct_storage_field(x, ::Val{I}) where {I} = getfield(x, I)

function _check_struct_storage(f::F, ::Type{T}, input) where {F,T}
    @test isbitstype(T)
    @test cuNumeric._struct_storage_type(T)
    output = f.(input)
    try
        @test output isa NDArray{T,1}
        for j in 1:fieldcount(T)
            # Complex-valued extraction still takes cuPyNumeric's unary path.
            fieldtype(T, j) <: Complex && continue
            field = _struct_storage_field.(output, Ref(Val(j)))
            try
                @test Array(field) == [getfield(f(Int64(i)), j) for i in 0:3]
            finally
                cuNumeric.destroy!(field)
            end
        end
        if T == StructStorageComplex
            for (projection, expected) in (
                (_struct_storage_real, Float32[1, 2, 3, 4]),
                (_struct_storage_imag, Float32[2, 3, 4, 5]),
            )
                field = projection.(output)
                try
                    @test Array(field) == expected
                finally
                    cuNumeric.destroy!(field)
                end
            end
        end
    finally
        cuNumeric.destroy!(output)
    end
end

@testset "Struct element storage" begin
    input = NDArray(Int64[0, 1, 2, 3])
    try
        _check_struct_storage(_struct_storage_float, StructStorageTriple{Float32}, input)
        _check_struct_storage(_struct_storage_uint, StructStorageTriple{UInt64}, input)
        _check_struct_storage(_struct_storage_packed, StructStoragePacked, input)
        _check_struct_storage(_struct_storage_padded, StructStoragePadded, input)
        _check_struct_storage(_struct_storage_complex, StructStorageComplex, input)
        _check_struct_storage(_struct_storage_named, typeof(_struct_storage_named(0)), input)

        @test fieldoffset(StructStoragePadded, 2) > sizeof(UInt8)
        @test isbitstype(StructStorageNested)
        @test isbitstype(StructStorageTupleField)
        @test_throws ArgumentError _struct_storage_nested.(input)
        @test_throws ArgumentError _struct_storage_tuple.(input)
        @test_throws ArgumentError similar(input, StructStorageNested, size(input))

        # Built-in complex arrays must keep Legate's numeric complex type.
        @test !cuNumeric._struct_storage_type(ComplexF32)
        complex_array = cuNumeric.zeros(ComplexF32, 4)
        try
            @test Array(complex_array) == zeros(ComplexF32, 4)
        finally
            cuNumeric.destroy!(complex_array)
        end
    finally
        cuNumeric.destroy!(input)
    end
end

@testset "Struct host transfer and in-place broadcast" begin
    initial = StructStorageParticle{Float32}[
        StructStorageParticle(0.0f0, 1.0f0),
        StructStorageParticle(2.0f0, -0.5f0),
    ]
    particles = NDArray(initial)
    try
        @test particles isa NDArray{StructStorageParticle{Float32},1}
        @test Array(particles) == initial
        @test occursin("StructStorageParticle", sprint(show, MIME"text/plain"(), particles))

        particles .= _struct_storage_step.(particles, 0.1f0)
        @test Array(particles) == StructStorageParticle{Float32}[
            StructStorageParticle(0.1f0, 1.0f0),
            StructStorageParticle(1.95f0, -0.5f0),
        ]
    finally
        cuNumeric.destroy!(particles)
    end

    mixed = reshape([_struct_storage_padded(i) for i in 0:5], 2, 3)
    device_mixed = NDArray(mixed)
    try
        @test Array(device_mixed) == mixed
    finally
        cuNumeric.destroy!(device_mixed)
    end
end

@testset "Struct storage round trips across layouts" begin
    for make_value in (
        _struct_storage_float,
        _struct_storage_packed,
        _struct_storage_padded,
        _struct_storage_complex,
        _struct_storage_named,
    )
        T = typeof(make_value(0))
        host = reshape(T[make_value(i) for i in 0:7], 2, 2, 2)
        device = NDArray(host)
        copied = similar(device)
        try
            @test size(device) == size(host)
            @test Array(device) == host
            @test Array{T,3}(device) == host
            @test copyto!(copied, device) === copied
            @test Array(copied) == host
        finally
            cuNumeric.destroy!(copied)
            cuNumeric.destroy!(device)
        end
    end
end

@testset "Preallocated struct storage needs no experimental opt-in" begin
    previous_experimental = get(task_local_storage(), :Experimental, false)
    input = NDArray(Int64[0, 1, 2, 3])
    output = similar(input, StructStoragePacked, size(input))
    empty_output = similar(input, StructStoragePacked, (0, 2))
    try
        cuNumeric.Experimental(false)
        @test output isa NDArray{StructStoragePacked,1}
        @test (output .= _struct_storage_packed.(input)) === output
        @test Array(output) == [_struct_storage_packed(i) for i in 0:3]
        @test size(empty_output) == (0, 2)
        @test isempty(Array(empty_output))
    finally
        cuNumeric.Experimental(previous_experimental)
        foreach(cuNumeric.destroy!, (input, output, empty_output))
    end
end

struct StructStorageRetagged
    x::Float32
    y::UInt8
    z::Int16
    w::UInt32
end
_struct_storage_retag(p::StructStoragePacked) = StructStorageRetagged(p.a, p.b, p.c, p.d)
_struct_storage_bump(p::StructStoragePadded, dx) = StructStoragePadded(p.a, p.b + dx, p.c, p.d)
function _struct_storage_add(p::StructStoragePadded, q::StructStoragePadded)
    return StructStoragePadded(p.a + q.a, p.b + q.b, p.c + q.c, p.d + q.d)
end
_struct_storage_padded_b(p::StructStoragePadded) = p.b
# Wrap indices so large arrays stay inside each field's range.
function _struct_storage_host(dims...)
    return reshape(
        [_struct_storage_padded(i % 200) for i in 0:(prod(dims) - 1)], dims...
    )
end

@testset "Struct storage across ranks" begin
    # Rank 0 cannot use the fused kernel, so transfers use fills and reshapes.
    scalar_host = fill(_struct_storage_padded(3))
    scalar = NDArray(scalar_host)
    scalar_copy = copy(scalar)
    try
        @test scalar isa NDArray{StructStoragePadded,0}
        @test Array(scalar) == scalar_host
        @test Array(scalar_copy) == scalar_host
        @test @allowscalar(scalar[]) == scalar_host[]
        @allowscalar scalar[] = _struct_storage_padded(9)
        @test Array(scalar)[] == _struct_storage_padded(9)
        @test Array(scalar_copy) == scalar_host
        @test !fetch(scalar == scalar_copy)
    finally
        foreach(cuNumeric.destroy!, (scalar, scalar_copy))
    end

    # Ranks above three pack struct stores through the generic dimension dispatch.
    for dims in ((5,), (3, 4), (2, 3, 4), (2, 1, 3, 2), (1, 2, 1, 2, 3))
        host = _struct_storage_host(dims...)
        device = NDArray(host)
        bumped = _struct_storage_bump.(device, 0.5)
        try
            @test Array(device) == host
            @test Array(bumped) == _struct_storage_bump.(host, 0.5)
        finally
            foreach(cuNumeric.destroy!, (device, bumped))
        end
    end

    # A multi-block launch covers the grid-stride loop for every thread.
    large_host = _struct_storage_host(64, 32, 17)
    large = NDArray(large_host)
    large_bumped = _struct_storage_bump.(large, 1.0)
    try
        @test Array(large_bumped) == _struct_storage_bump.(large_host, 1.0)
    finally
        foreach(cuNumeric.destroy!, (large, large_bumped))
    end
end

@testset "Struct copies, slices, and views" begin
    host = _struct_storage_host(4, 3)
    device = NDArray(host)
    try
        whole = copy(device)
        rows = device[2:3, :]
        rows_copy = copy(rows)
        try
            @test Array(whole) == host
            @test Array(rows) == host[2:3, :]
            @test Array(rows_copy) == host[2:3, :]
            @test Array(_struct_storage_bump.(rows, 1.0)) ==
                _struct_storage_bump.(host[2:3, :], 1.0)
            @test Array(permutedims(device)) == permutedims(host)
        finally
            foreach(cuNumeric.destroy!, (whole, rows, rows_copy))
        end

        # Slice destinations are transformed stores, so they copy with a kernel.
        replacement_host = reshape([_struct_storage_padded(100 + i) for i in 1:6], 2, 3)
        replacement = NDArray(replacement_host)
        try
            device[2:3, :] = replacement
            host[2:3, :] = replacement_host
            @test Array(device) == host
        finally
            cuNumeric.destroy!(replacement)
        end

        column = @view device[:, 2]
        column .= _struct_storage_bump.(column, 2.0)
        host[:, 2] .= _struct_storage_bump.(host[:, 2], 2.0)
        @test Array(device) == host

        mismatched = NDArray(_struct_storage_host(3, 4))
        try
            @test_throws DimensionMismatch copyto!(similar(device), mismatched)
        finally
            cuNumeric.destroy!(mismatched)
        end

        # Struct reshapes copy each field, following numeric reshape ordering.
        flat = cuNumeric.reshape(device, 3, 4)
        flat_b = cuNumeric.reshape(_struct_storage_padded_b.(device), 3, 4)
        try
            @test size(flat) == (3, 4)
            @test Array(_struct_storage_padded_b.(flat)) == Array(flat_b)
            @test_throws DimensionMismatch cuNumeric.reshape(device, 5, 2)
        finally
            foreach(cuNumeric.destroy!, (flat, flat_b))
        end
    finally
        cuNumeric.destroy!(device)
    end
end

@testset "Struct scalar indexing and fills" begin
    host = _struct_storage_host(3, 4)
    device = NDArray(host)
    filled = cuNumeric.fill(_struct_storage_padded(7), (2, 3))
    empty_filled = cuNumeric.fill(_struct_storage_padded(7), (0, 3))
    try
        @test_throws ErrorException device[2, 3]
        @test @allowscalar(device[2, 3]) == host[2, 3]
        @allowscalar device[3, 4] = _struct_storage_padded(42)
        host[3, 4] = _struct_storage_padded(42)
        @test Array(device) == host

        @test filled isa NDArray{StructStoragePadded,2}
        @test Array(filled) == fill(_struct_storage_padded(7), 2, 3)
        @test size(empty_filled) == (0, 3)
        @test fill!(device, _struct_storage_padded(1)) === device
        @test Array(device) == fill(_struct_storage_padded(1), 3, 4)
    finally
        foreach(cuNumeric.destroy!, (device, filled, empty_filled))
    end
end

@testset "Struct equality" begin
    host = _struct_storage_host(2, 3)
    changed_host = copy(host)
    changed_host[2, 2] = _struct_storage_padded(50)
    nan_host = [StructStorageTriple{Float32}(NaN32, 1, 2)]
    device, same, changed = NDArray(host), NDArray(host), NDArray(changed_host)
    nan_device = NDArray(nan_host)
    packed = NDArray([_struct_storage_packed(i) for i in 0:3])
    retagged = _struct_storage_retag.(packed)
    try
        @test fetch(device == same)
        @test !fetch(device == changed)
        @test fetch(device != changed)
        @test !fetch(device == NDArray(_struct_storage_host(3, 2)))
        # Records without a custom `==` compare with `===`, as in Base.
        @test fetch(nan_device == nan_device) == (nan_host == nan_host)
        # Identical layouts share a Legate type but are still different records.
        @test retagged isa NDArray{StructStorageRetagged,1}
        @test Array(retagged) == _struct_storage_retag.(Array(packed))
        @test fetch(packed == retagged) == (Array(packed) == Array(retagged))
    finally
        foreach(cuNumeric.destroy!, (device, same, changed, nan_device, packed, retagged))
    end
end

@testset "Struct broadcasts" begin
    host = _struct_storage_host(3, 2)
    device = NDArray(host)
    other = NDArray(reverse(host))
    empty_input = NDArray(Float32[])
    try
        sums = _struct_storage_add.(device, other)
        projected = _struct_storage_padded_b.(device)
        try
            @test Array(sums) == _struct_storage_add.(host, reverse(host))
            @test projected isa NDArray{Float64,2}
            @test Array(projected) == _struct_storage_padded_b.(host)
        finally
            foreach(cuNumeric.destroy!, (sums, projected))
        end
        empty_output = _struct_storage_float.(empty_input)
        @test empty_output isa NDArray{StructStorageTriple{Float32},1}
        @test isempty(Array(empty_output))
    finally
        foreach(cuNumeric.destroy!, (device, other, empty_input))
    end
end

@testset "Unsupported struct operations raise errors" begin
    host = _struct_storage_host(2, 3)
    device = NDArray(host)
    column = NDArray(reshape(Float64[1, 2], 2, 1))
    scalar = NDArray(fill(_struct_storage_padded(1)))
    try
        # Struct values cannot reach the kernel as broadcast scalars.
        @test_throws ArgumentError _struct_storage_add.(device, Ref(_struct_storage_padded(1)))
        # Struct results need the fused path: no size-1 extrusion and no rank 0.
        @test_throws ArgumentError _struct_storage_bump.(device, column)
        @test_throws ArgumentError _struct_storage_bump.(scalar, 1.0)
        shift = 1.0
        @test_throws ArgumentError (p -> _struct_storage_bump(p, shift)).(device)
        for reduction in (sum, prod, maximum, minimum)
            @test_throws ArgumentError reduction(device)
        end
        @test_throws ArgumentError sum(device; dims=1)
        # The host data survives each rejected operation.
        @test Array(device) == host
    finally
        foreach(cuNumeric.destroy!, (device, column, scalar))
    end
end
