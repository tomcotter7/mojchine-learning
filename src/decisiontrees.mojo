from matrix import Matrix
import math
from memory import UnsafePointer
from collections import Dict, Optional, Set
import random

from python import Python, PythonObject


fn _entropy(probabilities: List[Float64]) -> Float64:

  var t: Float64 = 0

  for prob in probabilities:
    t += (prob[] * math.log2(max(prob[], 1e-10)))
  
  return -1 * t

fn _gini(probabilities: List[Float64]) -> Float64:

  var t: Float64 = 0
  for prob in probabilities:
    t += (prob[] ** 2)

  return 1 - t

fn _get_percentile(owned data: List[Float64], percentile: Float64) raises -> Float64:
 
  sort(data)

  idx = (percentile * len(data)).__int__()

  return data[idx]

@value
struct Node:
    var feature_idx: Int
    var value: Float64
    var probs: List[Float64]
    var left: UnsafePointer[Node]
    var right: UnsafePointer[Node]

    fn __init__(
        out self,
        feature_idx: Int = -1,
        value: Float64 = math.inf[DType.float64](),
        probs: List[Float64] = List[Float64](),
        left: UnsafePointer[Node] = UnsafePointer[Node](),
        right: UnsafePointer[Node] = UnsafePointer[Node]()
    ):
        self.feature_idx = feature_idx
        self.value = value
        self.probs = probs
        self.left = left
        self.right = right

    fn __str__(self) -> String:
      s = "if data[" + self.feature_idx.__str__() + "] < " + self.value.__str__() + " then go left, else go right"
      return s

    fn is_leaf(self) -> Bool:

      if not self.left and not self.right:
        return True

      return False

fn free_node(node: Node) -> None:
  if node.left:
    free_node(node.left[])
    node.left.free()
  if node.right:
    free_node(node.right[])
    node.right.free()

@value
struct ClassificationTree():
  
  var criterion: String
  var max_depth: Int
  var min_samples_split: Int
  var max_features: String
  var min_information_gain: Float64
  var min_samples_leaf: Int
  var tree: Optional[Node]
  var unique_labels: List[Int]

  fn __init__(
      out self,
      criterion: String = "gini",
      max_features: String = "all",
      max_depth: Int = 10,
      min_samples_split: Int = 1,
      min_samples_leaf: Int = 2,
      min_information_gain: Float64 = 0.0,
      unique_labels: List[Int] = List[Int](0, 1),
      seed: Int = 420
    ):
    self.criterion = criterion
    self.max_features = max_features
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.min_information_gain = min_information_gain
    self.tree = None
    self.unique_labels = unique_labels
    self.min_samples_leaf = min_samples_leaf

    random.seed(seed)

  fn __del__(owned self):
    if self.tree:
      free_node(self.tree.value())

  fn get_class_index(self, class_value: Int) -> Int:
    for i in range(len(self.unique_labels)):
        if self.unique_labels[i] == class_value:
            return i
    return -1

  fn class_probabilties(self, labels: Matrix) raises -> List[Float64]:

    var probs = List[Float64](capacity=len(self.unique_labels))

    
    for _ in range(len(self.unique_labels)):
        probs.append(0)
    
    var total_count = labels.height
    
    for idx in range(labels.height):
        v = labels[idx, 0].__int__()
        class_idx = self.get_class_index(v)
        if class_idx >= 0:
            probs[class_idx] += 1
    
    for i in range(len(probs)):
        probs[i] = probs[i] / total_count
    
    return probs

  fn split_impurity(self, l_labels: Matrix, r_labels: Matrix) raises -> Float64:
    
    total_count = l_labels.height + r_labels.height

    var total_impurity: Float64 = 0

    f = _gini

    if self.criterion == "entropy":
      f = _entropy

    if l_labels.height > 0:
      total_impurity += (f(self.class_probabilties(l_labels)) * (l_labels.height / total_count))
    
    if r_labels.height > 0:
      total_impurity += (f(self.class_probabilties(r_labels)) * (r_labels.height / total_count))

    return total_impurity

  fn find_feature_idxs(self, total_features: Int) raises -> List[Int]:
    l = List[Int](capacity=total_features)
    for idx in range(total_features):
      l.append(idx)
    
    if self.max_features == "all":
      return l

    if self.max_features == "sqrt":
      random.shuffle(l)
      l = l[:math.sqrt(total_features)]
      return l
    
    raise Error("Max features mode: " + self.max_features + " is not available (yet)")

  fn find_split_points(self, col: Matrix) raises -> List[Float64]:

    values = col.to_list()
    sort(values)
    split_points = List[Float64]()
    
    for idx in range(len(values) - 1):
      if values[idx] != values[idx + 1]:
        midpoint = (values[idx] + values[idx + 1]) / 2
        split_points.append(midpoint)


    return split_points

  fn find_best_split(self, data: Matrix) raises -> Tuple[Float64, Int, Float64]:

    var min_split_impurity: Float64 = 1e6
    var min_impurity_feature_idx: Int = 0
    var min_impurity_feature_val: Float64 = 0

    idxs = self.find_feature_idxs(data.width - 1)

    for idx in idxs:
      col = data[":", idx[]]
      split_points = self.find_split_points(col)
      for feature_val in split_points:
        lsplit_idx = col.argwhere(col < feature_val[])
        rsplit_idx = col.argwhere(col >= feature_val[]) 
        l_data = data[lsplit_idx]
        r_data = data[rsplit_idx]

        if (l_data.height == 0) or (r_data.height == 0):
          continue

        split_impurity = self.split_impurity(l_data[":", -1], r_data[":", -1])
        if split_impurity < min_split_impurity:
          min_split_impurity = split_impurity
          min_impurity_feature_idx = idx[]
          min_impurity_feature_val = feature_val[]

    return min_impurity_feature_val, min_impurity_feature_idx, min_split_impurity

  fn create_tree(self, data: Matrix, current_depth: Int) raises -> Optional[Node]:

    if current_depth > self.max_depth:
      return None

    feature_val, feature_idx, split_entropy = self.find_best_split(data)

    labels = data[":", -1]
    label_probs = self.class_probabilties(labels) 
    
    if self.criterion == "entropy":
      node_entropy = _entropy(label_probs)
    else:
      node_entropy = _gini(label_probs)
    information_gain = node_entropy - split_entropy

    node = Node(feature_idx, feature_val, label_probs)

    col = data[":", feature_idx]
    lsplit_idx = col.argwhere(col < feature_val)
    rsplit_idx = col.argwhere(col >= feature_val) 
  
    l_data = data[lsplit_idx]
    r_data = data[rsplit_idx]

    if (l_data.height < self.min_samples_leaf) or (r_data.height < self.min_samples_leaf):
      return node

    if (self.min_samples_split > l_data.height) or (self.min_samples_split > r_data.height):
      return node

    if information_gain <= self.min_information_gain:
      return node


    lnode = self.create_tree(l_data, (current_depth + 1))
    if lnode:
      node.left = UnsafePointer[Node]().alloc(1)
      node.left.init_pointee_move(lnode.value())

    rnode = self.create_tree(r_data, (current_depth + 1))
    if rnode:
      node.right = UnsafePointer[Node]().alloc(1)
      node.right.init_pointee_move(rnode.value())

    return node

  fn train(mut self, data: Matrix) raises:
    unique_labels = Set[Int]()
    for idx in range(data.height):
      unique_labels.add(data[idx, -1].__int__())

    self.unique_labels = List[Int](capacity=unique_labels.__len__())

    for label in unique_labels:
      self.unique_labels.append(label[])
    
    self.tree = self.create_tree(data, 0)

  fn predict_one_sample(self, X: Matrix) raises -> List[Float64]:

    node = self.tree
    if not node:
      raise Error("Tree not initialized, call `train()` first.")

    while True:

      if node.value().is_leaf():
        return node.value().probs
      probs = node.value().probs
      
      if X[0, node.value().feature_idx] >= node.value().value:
        if not node.value().right:
          return probs
        node = Optional[Node](node.value().right[])
      else:
        if not node.value().left:
          return probs
        node = Optional[Node](node.value().left[])

  fn predict_proba(self, X: Matrix) raises -> Matrix:

    pred_probs = List[List[Float64]](capacity=X.height)

    for i in range(X.height):
      row = X[i]
      probs = self.predict_one_sample(row)
      pred_probs.append(probs)

    
    m = Matrix.from_list(pred_probs)
    return m

  fn predict(self, X: Matrix) raises -> List[Float64]:

    pred_probs = self.predict_proba(X)
    preds = pred_probs.argmax(axis=1)

    return preds

fn parse_into_list(dataset_features: PythonObject, dataset_labels: PythonObject) raises -> Matrix:

  ds = List[List[Float64]](capacity=len(dataset_features))

  for idx in range(len(dataset_features)):

    row = List[Float64](capacity=len(dataset_features[idx]) + 1)
    X = dataset_features[idx]
    y = dataset_labels[idx]

    for row_idx in range(len(X)):
      value = X[row_idx]
      row.append(Float64(value))

    row.append(Float64(y))

    ds.append(row)

  df = Matrix.from_list(ds)
  return df

fn predict(dt: ClassificationTree, dataset: Matrix) raises -> Float64:
  pos = dataset.width - 1
  features = dataset[":", 0:pos]

  y_true = dataset[":", -1]
  y_pred = dt.predict(features)

  correct = 0

  for idx in range(len(y_pred)):

    val_pred = y_pred[idx]
    val_true = y_true[idx, 0]
    
    if val_pred == val_true:
      correct += 1

  accuracy = correct / len(y_pred)

  return accuracy

fn main() raises:

  datasets = Python.import_module("sklearn.datasets")
  model_selection = Python.import_module("sklearn.model_selection")

  data = datasets.load_iris(return_X_y=True)

  X = data[0]
  y = data[1]

  r = model_selection.train_test_split(X, y, test_size=0.2, random_state=420)
  train_data = parse_into_list(r[0], r[2])
  test_data = parse_into_list(r[1], r[3])

  dt = ClassificationTree(criterion="gini", max_features="sqrt")
  dt.train(train_data)

  accuracy = predict(dt, train_data)

  print("accuracy on train set:", accuracy)

  accuracy = predict(dt, test_data)

  print("accuracy on test set: ", accuracy)



