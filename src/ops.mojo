@always_inline
fn add[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return a + b

@always_inline
fn scalar_add[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, 1]) -> SIMD[dtype, width]:
    return a + b

@always_inline
fn scalar_mul[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, 1]) -> SIMD[dtype, width]:
    return a * b

@always_inline
fn scalar_sub[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, 1]) -> SIMD[dtype, width]:
    return a - b

@always_inline
fn scalar_pow[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, 1]) -> SIMD[dtype, width]:
    return a ** b

@always_inline
fn scalar_truediv[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, 1]) -> SIMD[dtype, width]:
    return a / b

@always_inline
fn mul[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
  return a * b

@always_inline
fn sub[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
  return a - b
