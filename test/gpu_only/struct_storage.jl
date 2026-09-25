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
