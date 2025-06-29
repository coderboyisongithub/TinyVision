from py_loader import pytt as tt
class UnifiedMeta(type(tt.nn.Module), type):
    pass

class AutoRegisterMeta(UnifiedMeta):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)

        # 自动收集所有子模块
        children = []
        for name in dir(instance):
            if name.startswith('_'):
                continue
            attr = getattr(instance, name)
            if isinstance(attr, tt.nn.Module):
                children.append(attr)

        # 注册所有子模块
        if children:
            instance.registerModules(children)

        return instance
class Module(tt.nn.Module, metaclass=AutoRegisterMeta):
    def __init__(self):
        super().__init__()

    def __setattr__(self, name, value):
        """重写 __setattr__ 以支持动态添加子模块"""
        super().__setattr__(name, value)

        # 如果是新添加的模块，立即注册
        if isinstance(value, tt.nn.Module):
            self.registerModules([value])

    def add_module(self, name, module):
        """官方推荐的模块注册方式"""
        self._modules[name] = module
        self.registerModules([module])
        setattr(self, name, module)  # 设置属性

    def modules(self):
        """返回所有子模块的迭代器"""
        return self._modules.values()

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