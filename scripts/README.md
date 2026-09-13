## Scripts

After initializing the git submodules, we need to install them.
- `install_cxxwrap.sh` WARNING: This will overwrite `/home/user/.julia/dev/libcxxwrap_julia_jll/override`.





We struggled compiling the wrapper package with g++ `(Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0` and CUDA `cuda_12.3.r12.3/compiler.33567101_0`.
The error is shown below. This seems to have mismatch with the atomic ref C++ versioning; however, we weren't able to pinpoint the exact issue. We have patched legion with `legion_redop.inl`. This is a temporary "hack" solution.

```
In file included from /usr/include/c++/11/bits/shared_ptr_atomic.h:33,
                 from /usr/include/c++/11/memory:78,
                 from /home/david/cuNumeric.jl/libcxxwrap-julia/include/jlcxx/jlcxx.hpp:7,
                 from /home/david/cuNumeric.jl/wrapper/wrapper.cpp:1:
/usr/include/c++/11/bits/atomic_base.h: In instantiation of ‘struct std::__atomic_ref<cuda::std::__4::complex<__half>, false, false>’:
/usr/include/c++/11/atomic:1668:12:   required from ‘struct std::atomic_ref<cuda::std::__4::complex<__half> >’
/home/david/anaconda3/envs/cupy_gpu/include/legion/legion_redop.inl:1380:36:   required from here
/usr/include/c++/11/bits/atomic_base.h:1337:21: error: static assertion failed
 1337 |       static_assert(is_trivially_copyable_v<_Tp>);
      |                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
/usr/include/c++/11/bits/atomic_base.h:1337:21: note: ‘std::is_trivially_copyable_v<cuda::std::__4::complex<__half> >’ evaluates to false
gmake[2]: *** [CMakeFiles/cupynumericwrapper.dir/build.make:79: CMakeFiles/cupynumericwrapper.dir/wrapper/wrapper.cpp.o] Error 1
gmake[2]: Leaving directory '/home/david/cuNumeric.jl/build'
gmake[1]: *** [CMakeFiles/Makefile2:87: CMakeFiles/cupynumericwrapper.dir/all] Error 2
gmake[1]: Leaving directory '/home/david/cuNumeric.jl/build'
gmake: *** [Makefile:136: all] Error 2

```


This error was shown for each reduction operator in `legion_redop.inl` for complex types. The unmodified code of the Sum reduction is below. We would enter `#if defined(__cpp_lib_atomic_ref) && (__cpp_lib_atomic_ref >= 201806L)`; however, the compiler failed the static assert when `std::atomic_ref<LHS> atomic(lhs);` was used. We have added a patch `patch_legion.sh` for the various default reductions where it does the TypePunning case. This will copy our patched file into the conda installed path of cupynumeric/legion.

```
#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
  template<> __CUDA_HD__ inline
  void SumReduction<complex<__half> >::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<complex<__half> >::apply<false>(LHS &lhs, RHS rhs)
  {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    RHS newval = lhs, oldval;
    // Type punning like this is illegal in C++ but the
    // CUDA manual has an example just like it so fuck it
    unsigned int *ptr = (unsigned int*)&lhs;
    do {
      oldval = newval;
      newval += rhs;
      newval = __uint_as_complex(atomicCAS(ptr,
            __complex_as_uint(oldval), __complex_as_uint(newval)));
    } while (oldval != newval);
#else
#if defined(__cpp_lib_atomic_ref) && (__cpp_lib_atomic_ref >= 201806L)
    std::atomic_ref<LHS> atomic(lhs);
    RHS oldval = atomic.load();
    RHS newval;
    do {
      newval = oldval + rhs;
    } while (!atomic.compare_exchange_weak(oldval, newval));
#else
    TypePunning::Alias<int32_t,complex<__half> > oldval, newval;
    TypePunning::Pointer<int32_t> pointer((void*)&lhs);
    do {
      oldval.load(pointer);
      newval = oldval.as_two() + rhs;
    } while (!__sync_bool_compare_and_swap((int32_t*)pointer,
                      oldval.as_one(), newval.as_one()));
#endif
#endif
```
