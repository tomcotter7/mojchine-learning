from memory import UnsafePointer, memcpy, memset_zero
from algorithm import vectorize, parallelize
from sys import simdwidthof
from ops import (
  add, scalar_add, scalar_mul, scalar_sub,
  scalar_pow, scalar_truediv, mul, sub
)

struct Matrix(Stringable, Writable):
  var height: Int
  var width: Int
  var size: Int
  var data: UnsafePointer[Float64]


  alias simdwidth = simdwidthof[DType.float64]()

  @always_inline
  fn __init__(out self, height: Int, width: Int):
    self.width = width
    self.height = height
    self.size = width * height
    self.data = UnsafePointer[Float64].alloc(self.size)

  @always_inline
  fn __init__(out self, height: Int, width: Int, data: UnsafePointer[Float64]):
    self.width = width
    self.height = height
    self.size = width * height
    self.data = data
  
  @always_inline
  fn __init__(out self, matrix: String) raises:
    var mat = matrix.replace(" ", "")
    if mat[0] == '[' and mat[1] == '[' and mat[len(mat) - 1] == ']' and mat[len(mat) - 2] == ']':
      self.width = 0
      self.size = 0
      self.data = UnsafePointer[Float64]()
      var rows = mat[:-1].split(']')
      self.height = len(rows) - 1
      for i in range(self.height):
        var values = rows[i][2:].split(',')
        if i == 0:
          self.width = len(values)
          self.size = self.height * self.width
          self.data = UnsafePointer[Float64].alloc(self.size)
        for j in range(self.width):
          self.store(i, j, atof(values[j]).cast[DType.float64]())

    else:
      raise Error('Error: Matrix is not initialized in the correct form!')

  @staticmethod
  fn from_list(data: List[List[Float64]]) raises -> Matrix:

    m = Matrix(height=len(data), width=len(data[0]))

    for i in range(len(data)):
      for j in range(len(data[i])):
        m[i, j] = data[i][j]

    return m^

  fn __copyinit__(out self, existing: Matrix):
    self.width = existing.width
    self.height = existing.height
    self.size = self.width * self.height
    self.data = UnsafePointer[Float64]().alloc(self.size)
    memcpy(self.data, existing.data, self.size)

  fn __del__(owned self):
    if self.data:
      self.data.free()

  
  fn __getitem__(self, row: Int, col: Int) raises -> Float64:
    if row >= self.height or col >= self.width:
      var err: String = "OOB Error: Matrix has shape:" + self.shape() + ". Attempted to access (row, col): (" + row.__str__() + "," + col.__str__() + ")"
      raise Error(err)
    idx = (row * self.width) + col
    return self.data[idx]

  fn __getitem__(self, row: Int) raises -> Matrix:
    
    if row >= self.height:
      var err: String = "OOB Error: Matrix has shape:" + self.shape() + ". Attempted to access row: " + row.__str__()
      raise Error(err)

    m = Matrix(1, self.width)
    memset_zero(m.data, m.size)

    @parameter
    fn build[nelts: Int](i: Int):
      m.store[nelts](0, i, self.load[nelts](row, i))

    vectorize[build, self.simdwidth](self.width)

    return m^

  fn __getitem__(self, row: String, owned col: Int) raises -> Matrix:
    
    if col >= self.width:
      var err: String = "OOB Error: Matrix has shape:" + self.shape() + ". Attempted to access col: " + col.__str__()
      raise Error(err)

    if col < 0:
      col = self.width + col

    
    m = Matrix(self.height, 1)
    memset_zero(m.data, m.size)
  
    @parameter
    fn build[nelts: Int](i: Int):
      m.store[nelts](i, 0, self.strided_load[nelts](i, col, self.width))

    vectorize[build, self.simdwidth](self.height)
   
    return m^

  fn __getitem__(self, rows: Slice) raises -> Matrix:
    
    end = rows.end.or_else(self.height)
    start = rows.start.or_else(0)

    if end >= self.height or start < 0:
      var err: String = "OOB Error: Matrix has shape:" + self.shape() + ". Attempted to access rows: " + rows.__str__()
      raise Error(err)

    height = (end - start)

    m = Matrix(height, self.width)
    memset_zero(m.data, m.size)
   
    for i in range(start, end, 1):
      for j in range(self.width):
        m[i - start, j] = self[i, j]
      
    return m^

  fn __getitem__(self, rows: String, cols: Slice) raises -> Matrix:

    end = cols.end.or_else(self.width)
    start = cols.start.or_else(0)
    width = (end - start)

    if end >= self.width or start < 0:
      var err: String = "OOB Error: Matrix has shape:" + self.shape() + ". Attempted to access cols: " + cols.__str__()
      raise Error(err)

    m = Matrix(self.height, width)
    memset_zero(m.data, m.size)

    for i in range(self.height):
      for j in range(start, end, 1):
        var val: Float64 = self[i, j]
        m[i, j - start] = val
    
    return m^

  fn __getitem__(self, rows: List[Int]) raises -> Matrix:

    m = Matrix(len(rows), self.width)

    for row_idx in range(len(rows)):
      m.__setitem__(row_idx, self[rows[row_idx]])

    return m^

  fn __setitem__(self, row: Int, col: Int, item: Float64) raises:
    if row > self.height or row < 0 or col > self.width or col < 0:
      var err: String = "OOB Error: Matrix has shape:" + self.shape() + ". Tried to put value into index: (" + row.__str__() + "," + col.__str__() + ")"
      raise Error(err)
    idx = (row * self.width) + col
    self.data[idx] = item

  fn __setitem__(self, row_idx: Int, row: Matrix) raises:
    if row_idx > self.height:
      var err: String = "OOB Error: Matrix has shape:" + self.shape() + ". Tried to copy a row into index: " + row_idx.__str__()
      raise Error(err)
    idx = (row_idx * self.width)
    src = self.data + idx
    memcpy(src, row.data, row.width)

  @always_inline
  fn strided_load[width: Int](self, row: Int, col: Int, stride: Int) -> SIMD[DType.float64, width]:
    idx = (row * self.width) + col
    tmpPtr = self.data + idx
    return tmpPtr.strided_load[width=width](stride)
  
  
  @always_inline
  fn load[width: Int](self, row: Int, col: Int) -> SIMD[DType.float64, width]:
    idx = (row * self.width) + col
    return self.data.load[width=width](idx)
    
  @always_inline
  fn store[width: Int](self, row: Int, col: Int, value: SIMD[DType.float64, width]):
    idx = (row * self.width) + col
    self.data.store(idx, value)

  fn _elemwise_operation[func: fn[dtype: DType, width: Int](SIMD[dtype, width], SIMD[dtype, width]) -> SIMD[dtype, width]](self, rhs: Matrix) -> Matrix:
    m = Matrix(self.height, self.width, UnsafePointer[Float64].alloc(self.size))
    @parameter
    fn op[simd_width: Int](idx: Int):
      m.data.store(idx, func[DType.float64, simd_width](self.data.load[width=simd_width](idx), rhs.data.load[width=simd_width](idx)))
    vectorize[op, self.simdwidth](self.size)
    return m^

  fn _elemwise_scalar_operation[func: fn[dtype: DType, width: Int](SIMD[dtype, width], SIMD[dtype, 1]) -> SIMD[dtype, width]](self, rhs: Float64) -> Matrix:
    m = Matrix(self.height, self.width, UnsafePointer[Float64].alloc(self.size))
    @parameter
    fn op[simd_width: Int](idx: Int):
      m.data.store(idx, func[DType.float64, simd_width](self.data.load[width=simd_width](idx), rhs))
    vectorize[op, self.simdwidth](self.size)
    return m^
      
  fn __add__(self, rhs: Matrix) -> Matrix:
    return self._elemwise_operation[add](rhs)

  fn __add__(self, rhs: Float64) -> Matrix:
    return self._elemwise_scalar_operation[scalar_add](rhs)

  fn __mul__(self, rhs: Matrix) -> Matrix:
    return self._elemwise_operation[mul](rhs)

  fn __mul__(self, rhs: Float64) -> Matrix:
    return self._elemwise_scalar_operation[scalar_mul](rhs)

  fn __sub__(self, rhs: Matrix) -> Matrix:
    return self._elemwise_operation[sub](rhs)

  fn __sub__(self, rhs: Float64) -> Matrix:
    return self._elemwise_scalar_operation[scalar_sub](rhs)

  fn __rsub__(self, lhs: Float64) raises -> Matrix: 
    m = Matrix(self.height, self.width, UnsafePointer[Float64].alloc(self.size))
    
    for i in range(self.height):
      for j in range(self.width):
        m[i, j] = (1 - self[i, j])

    return m^

  fn __pow__(self, rhs: Float64) -> Matrix:

    return self._elemwise_scalar_operation[scalar_pow](rhs)

  fn __truediv__(self, rhs: Float64) -> Matrix:
    
    return self._elemwise_scalar_operation[scalar_truediv](rhs)

  fn __matmul__(self, owned rhs: Matrix) raises -> Matrix:

    if self.width != rhs.height:
      raise Error("Matrix dimension do not match!")

    m = Matrix(self.height, rhs.width, UnsafePointer[Float64].alloc(self.height * rhs.width))
    memset_zero(m.data, m.size)

    rhs = rhs.T()
    
    tile_size = 64
  
    for i_tile in range(0, self.height, tile_size):
      i_end = min(i_tile + tile_size, self.height)
      for j_tile in range(0, rhs.height, tile_size):
        j_end = min(j_tile + tile_size, rhs.height)
        for i in range(i_tile, i_end):
          for j in range(j_tile, j_end):
            @parameter
            fn dotproduct[nelts: Int](k: Int):
              row = self.load[nelts](i, k)
              col = rhs.load[nelts](j, k)
              m.store[1](i, j, m.load[1](i, j) + (row * col).reduce_add())

            vectorize[dotproduct, self.simdwidth](self.width)

    return m^
  
  @always_inline
  fn __le__(self, rhs: Float64) -> List[Bool]:
    var result = List[Bool](capacity=self.size)

    for i in range(self.size):
      result[i] = self.data[i] <= rhs

    return result^
  
  @always_inline
  fn __ge__(self, rhs: Float64) -> List[Bool]:
    var result = List[Bool](capacity=self.size)

    for i in range(self.size):
      result[i] = self.data[i] >= rhs

    return result^

  @always_inline
  fn __gt__(self, rhs: Float64) -> List[Bool]:
    var result = List[Bool](capacity=self.size)

    for i in range(self.size):
      result[i] = self.data[i] > rhs

    return result^

  @always_inline
  fn __lt__(self, rhs: Float64) -> List[Bool]:
    var result = List[Bool](capacity=self.size)

    for i in range(self.size):
      result[i] = self.data[i] < rhs

    return result^

  fn to_list(self) raises -> List[Float64]:
    
    m = self
    if m.height != 1:
      m = m.T()

    if m.height != 1:
      raise Error("Cannot transform a non 1-D Matrix to a List")

    x = List[Float64](capacity=m.width)

    for idx in range(m.width):
      x.append(m[0, idx])

    return x^

  fn sum(self) -> Float64:

    total = 0.0
    
    for i in range(self.height):
      @parameter
      fn sum[width: Int](j: Int):
        total += self.load[width](i, j).reduce_add[1]()[0]
      vectorize[sum, self.simdwidth](self.width)

    return total

  fn mean(self) -> Float64:

    total = 0.0
    
    for i in range(self.height):
      @parameter
      fn sum[width: Int](j: Int):
        total += self.load[width](i, j).reduce_add[1]()[0]
      vectorize[sum, self.simdwidth](self.width)

    return total / self.size

  fn write_to[W: Writer](self, mut writer: W):
    
    s = String("[")
    
    for i in range(self.height):
      s += "["
      for j in range(self.width):
        s += self.load[1](i, j).__str__()
        if j < self.width - 1:
          s += ","

      s += "]"
      if i < self.height - 1:
        s += ",\n"

    s += "]"
    writer.write(s)

  fn __str__(self) -> String:
    return String.write(self)

  fn shape(self) -> String:
    return "(" + self.height.__str__() + "," + self.width.__str__() + ")"
  
  @always_inline
  fn T(self) raises -> Matrix:
    m = Matrix(self.width, self.height, UnsafePointer[Float64].alloc(self.size))
    for i in range(self.height):
      for j in range(self.width):
        m[j, i] = self[i, j]

    return m^

  fn sign(self) raises -> Matrix:
    m = Matrix(self.height, self.width, UnsafePointer[Float64].alloc(self.size))
    for i in range(self.height):
      for j in range(self.width):
        v = self[i, j]
        if v > 0:
          m[i, j] = 1
        elif v < 0:
          m[i, j] = -1
        else:
          m[i, j] = 0

    return m^

  fn argwhere(self, cmp: List[Bool]) -> List[Int]:
    var args = List[Int]()
    for i in range(self.size):
      if cmp[i]:
        args.append(i)

    return args^

  fn argmax(self, axis: Int) raises -> List[Float64]:
    if axis == 1:
      max_values = List[Float64](capacity=self.height)
      for i in range(self.height):
        current_max = -1e-6
        current_idx = 0
        for j in range(self.width):
          v = self[i, j]
          if v > current_max:
            current_max = v 
            current_idx = j
        max_values.append(current_idx)

      return max_values

    elif axis == 0:
      max_values = List[Float64](capacity=self.height)
      for j in range(self.width):
        current_max = -1e-6
        current_idx = 0
        for i in range(self.height):
          v = self[i, j]
          if v > current_max:
            current_max = v
            current_idx = i

        max_values.append(current_idx)

      return max_values

    raise Error("Incorrect Axis")


          

          

        

    

