from gpu import thread_idx
from gpu.host import DeviceContext


def main():

  fn printing_kernel():
      print("GPU thread: [", thread_idx.x, thread_idx.y, thread_idx.z, "]")

  var ctx = DeviceContext()

  ctx.enqueue_function[printing_kernel](grid_dim=1, block_dim=4)
  ctx.synchronize()
