import numpy

from py_loader import pytt as tt

class UnifiedMeta(type(tt.nn.Module), type):
    pass

class AutoRegisterMeta(UnifiedMeta):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        #instance.set_name(instance.get_class_name())
        # 自动收集所有子模块
        instance.set_name("")
        module_entries = []
        for name, attr in vars(instance).items():  # 直接遍历__dict__
            if name.startswith('_'):
                continue
            if isinstance(attr, tt.nn.Module):
                # 设置子模块名称：格式为"父模块名.变量名"
               # attr.set_name(f"{instance.name()}.{name}")
                module_entries.append(attr)

        # 批量注册子模块
        if module_entries:
            instance.registerModules(module_entries)

        cls._recursive_register(instance, instance.name())
        return instance

    @classmethod
    def _recursive_register(self, module, parent_name):
        """递归注册子模块并设置层级化名称"""

        # 特殊处理Sequential的子模块（通过__iter__获取）
        if isinstance(module, tt.nn.Sequential):
            for idx, child in enumerate(module):  # 使用__iter__
                if not hasattr(child, 'name'):
                    continue
                # Sequential的子模块命名为：父名.[序号]
                new_name = f"{idx}"
                child.set_name(new_name)
                self._recursive_register(child, new_name)

        if not hasattr(module, '__dict__'):
            return

        # 处理当前模块的直接子模块
        for name, attr in vars(module).items():
            if name.startswith('_'):
                continue
            if isinstance(attr, tt.nn.Module):
                # 设置子模块名称：父名.变量名
                if parent_name == "":
                    new_name = f"{name}"
                else:
                    new_name = f"{parent_name}.{name}"
                attr.set_name(new_name)
                # 递归处理子模块的子模块
                self._recursive_register(attr, new_name)

class Module(tt.nn.Module, metaclass=AutoRegisterMeta):
    def __init__(self):
        super().__init__()

    def load_state_dict(self, state_dict: dict, device: str = 'cpu'):
        # 类型检查
        if not isinstance(state_dict, dict):
            raise TypeError("state_dict must be a dictionary")

        # 构建C++兼容输入
        cpp_dict = {}
        for name, tensor in state_dict.items():
            if not isinstance(name, str):
                raise TypeError(f"Key must be string, got {type(name)}")
            if isinstance(tensor, numpy.ndarray):
                cpp_dict[name] = tt.Tensor(tensor)
            elif isinstance(tensor, tt.Tensor):
                cpp_dict[name] = tensor
            else:
                raise TypeError(f"Unsupported tensor type: {type(tensor)}")
        # 调用C++
        try:
            self.load(cpp_dict, device)
        except RuntimeError as e:
            raise ValueError(f"Failed to load state_dict: {str(e)}")

    def get_class_name(self):
        return self.__class__.__name__

    def modules(self):
        """返回所有子模块的迭代器"""
        return self._modules.values()

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        """统一入口点，根据输入类型自动路由"""
        # 1. 单 Tensor 输入
        if len(args) == 1 and isinstance(args[0], tt.Tensor):
            return self.forward_single(args[0])

        # 2. Tensor 列表输入
        if len(args) == 1 and isinstance(args[0], list) and all(isinstance(x, tt.Tensor) for x in args[0]):
            return self.forward_multi_input(args[0])

        # 3. 多个 Tensor 作为独立参数
        if len(args) > 1 and all(isinstance(x, tt.Tensor) for x in args):
            return self.forward_multi_input(list(args))

        # 4. 特殊处理：单输入返回多输出
        if len(args) == 1 and hasattr(self, 'forward_single_to_multi'):
            return self.forward_single_to_multi(args[0])

        raise TypeError(f"Unsupported input types: {[type(a) for a in args]}")

    # ---------- 默认实现 ----------
    def forward_single(self, x: tt.Tensor) -> tt.Tensor:
        """单输入 -> 单输出 (默认实现)"""
        return x

    def forward_multi_input(self, inputs: list[tt.Tensor]) -> tt.Tensor:
        """多输入 -> 单输出 (默认实现)"""
        return inputs[0]  # 返回第一个输入

    def forward_single_to_multi(self, x: tt.Tensor) -> list[tt.Tensor]:
        """单输入 -> 多输出 (需要时实现)"""
        return [x]  # 默认返回单元素列表