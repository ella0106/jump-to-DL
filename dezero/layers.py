import numpy as np
import dezero.functions as F
from dezero.core import Parameter
import weakref

class Layer:
    def __init__(self) -> None:
        self._params = set()

    def __setattr__(self, name: str, value) -> None:
        if isinstance(value, Parameter):
            self._params.add(name)

        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            yield self.__dict__[name]
            # yield 는 return과 같지만, 처리를 종료하고 값을 반환하는 것이 아니라 작업을 중단하다가 재개함
            # 그래서 아래 cleargrads에서 self.params()를 호출 할 때 다시 재개하여 처음부터 불러오는 것이 아닌 순차적으로 반환이 가능하다는 것이다

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None: #in_size가 있으면 W 초기화
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')
        
    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            # in_size가 없을 때 forward 시점에서 크기 조정
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y