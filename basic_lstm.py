import theano
import theano.tensor as T
from layers import LSTMLayer, InputLayer, FullyConnectedLayer
from lib import get_params, make_caches, SGD,\
        momentum, create_optimization_updates


class basicLSTM():

    def __init__(self, num_input, num_cells=50, num_output=1):
        X = T.matrix('x')
        Y = T.matrix('y')
        eta = T.scalar('eta')
        alpha = T.scalar('alpha')

        self.num_input = num_input
        self.num_output = num_output
        self.num_cells = num_cells
        self.eta = eta

        inputs = InputLayer(X, name="inputs")
        lstm = LSTMLayer(num_input, num_cells, input_layer=inputs, name="lstm")
        fc = FullyConnectedLayer(num_cells, num_output, input_layer=lstm)
        Y_hat = T.mean(fc.output(), axis=2)
        layer = inputs, lstm, fc
        self.params = get_params(layer)
        self.caches = make_caches(self.params)
        self.layers = layer
        mean_cost = T.mean((Y - Y_hat)**2)
        last_cost = T.mean((Y[-1] - Y_hat[-1])**2)
        self.cost = alpha*mean_cost + (1-alpha)*last_cost
        """"
        self.updates = momentum(self.cost, self.params, self.caches, self.eta, clip_at=3.0)
        """
        self.updates,_,_,_,_ = create_optimization_updates(self.cost, self.params, method="adadelta")
        self.train = theano.function([X, Y, alpha], [self.cost, last_cost] ,\
                updates=self.updates, allow_input_downcast=True)
        self.costfn = theano.function([X, Y, alpha], [self.cost, last_cost],\
                allow_input_downcast=True)
        self.predict = theano.function([X], [Y_hat], allow_input_downcast=True)


    def reset_state(self):
        for layer in self.layers:
            layer.reset_state() 
