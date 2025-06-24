from py_loader import pytt as tt

class Module(tt.nn.Module):
    def __init__(self):
        super().__init__()
        # 我们将在__setattr__中识别模块并自动注册
        self._modules = {}  # 用于存储模块，以便于管理

    def __setattr__(self, name, value):
        # 首先设置属性
        super().__setattr__(name, value)
        # 如果value是Module类型，则自动注册
        if isinstance(value, Module):
            # 将模块添加到内部字典（类似于PyTorch的机制）
            self._modules[name] = value
            # 调用C++的add_module方法，将模块添加到C++管理的children_中
            self.add_module(value)

    def modules(self):
        """返回所有子模块的迭代器"""
        return self._modules.values()