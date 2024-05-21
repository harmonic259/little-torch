from rsdl.optim import Optimizer

# TODO: implement step function
class SGD(Optimizer):
    def __init__(self, layers, learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        # TODO: update weight and biases ( Don't use '-=' and use l.weight = l.weight - ... )
        for l in self.layers:
            l.weight.data = l.weight.data - self.learning_rate * l.weight.grad.data
            if l.need_bias:
                l.bias.data = l.bias.data - self.learning_rate * l.bias.grad.data
