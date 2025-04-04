from matrix import Matrix
from memory import UnsafePointer
from math import exp, log

fn sigmoid(ipt: Float64) -> Float64:
  
  exponential = exp(ipt)
  return (exponential / (1 + exponential))

fn sigmoid(ipt: Matrix) raises -> Matrix:

  m = Matrix(ipt.height, ipt.width, UnsafePointer[Float64].alloc(ipt.size))

  for i in range(ipt.height):
    for j in range(ipt.width):
      m[i, j] = sigmoid(ipt[i, j])

  return m

fn mlog(ipt: Matrix) raises -> Matrix:

  m = Matrix(ipt.height, ipt.width, UnsafePointer[Float64].alloc(ipt.size))

  for i in range(ipt.height):
    for j in range(ipt.width):
      value = ipt[i, j]
      if value < 0:
        m[i, j] = log(min(value, -1e-15))
      else:
        m[i, j] = log(max(value, 1e-15))

  return m^
  
@value
struct LogisticRegression():
  
  var weights: Matrix
  var bias: Float64
  var lr: Float64
  var threshold: Float64

  fn _backward(mut self, x: Matrix, y: Matrix) raises:
    y_pred = self._forward(x)

    xt = x.T()
    diff = (y_pred - y)
    
    r = xt @ diff
    dldw =  r * (1 / x.height)
    dldb = (diff).sum() * (1/ x.height)

    self.weights = self.weights - (dldw * self.lr)
    self.bias = self.bias - (dldb * self.lr)

  fn _forward(self, x: Matrix) raises -> Matrix:  
    return sigmoid((x @ self.weights) + self.bias)

  fn fit(mut self, x: Matrix, y: Matrix, epochs: Int) raises:
    
    prev_loss = 0.0
    for epoch in range(epochs):
      y_pred = self._forward(x)
      self._backward(x, y)
      if epoch % 5 == 0:
        y_pred = self._forward(x)
        loss = ((y * mlog(y_pred) + (1 - y) * mlog(1 - y_pred)) * -1).mean()
        print("Epoch: ", epoch, "Loss: ", loss)
        if abs(loss - prev_loss) < self.threshold:
          print("Early stopping on epoch:", epoch)
          return
        prev_loss = loss

fn main() raises:

    X = Matrix("[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]")
    y = Matrix("[[0.0], [1.0], [1.0], [1.0]]")

    
    model = LogisticRegression(
        weights=Matrix("[[0.0], [0.0]]"),
        bias=0.0,
        lr=0.01,
        threshold=0.000001
    )
    
    model.fit(X, y, epochs=10000)
