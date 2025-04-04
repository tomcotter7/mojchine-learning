from matrix import Matrix

@value
struct LinearRegression():
  
  var weights: Matrix
  var bias: Float64
  var lr: Float64
  var l1: Float64
  var threshold: Float64

  fn _backward(mut self, x: Matrix, y: Matrix) raises:
    y_pred = self._forward(x) 
    xT = x.T()

    dldw = ((xT @ (y - y_pred)) * -2) / x.height
    dldw = dldw + (self.weights.sign() * self.l1)

    dldb = ((y - y_pred).sum() * -2) / x.height

    self.weights = self.weights - (dldw * self.lr)
    self.bias = self.bias - (dldb * self.lr)

  fn _forward(self, x: Matrix) raises -> Matrix:
    return (x @ self.weights) + self.bias
     
  fn fit(mut self, x: Matrix, y: Matrix, epochs: Int) raises:
    
    prev_loss = 0.0
    for epoch in range(epochs):
      self._backward(x, y)
      if epoch % 5 == 0:
        loss = ((y - self._forward(x)) ** 2).sum()
        print("Epoch: ", epoch, "Loss: ", loss)
        if abs(loss - prev_loss) < self.threshold:
          print("Early stopping on epoch:", epoch)
          return
        prev_loss = loss

  
  fn predict(self, x: Matrix) raises -> Matrix:
    return self._forward(x)

fn main() raises:
    X = Matrix("[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]")
    
    y = Matrix("[[2.0], [4.0], [6.0]]")
    
    model = LinearRegression(
        weights=Matrix("[[0.1], [0.1]]"),
        bias=0.0,
        lr=0.001,
        l1=0.001,
        threshold=0.000001
    )
    
    model.fit(X, y, epochs=100)
