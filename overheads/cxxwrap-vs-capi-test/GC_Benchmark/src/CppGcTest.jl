module CppGcTest

using CxxWrap

lib = "libcxxwrapgctest.so"
@wrapmodule(() -> "/home/david/cuNumeric.jl/examples/test/cpp/build/lib/libcxxwrapgctest.so")


function __init__()
    @initcxx
end

end