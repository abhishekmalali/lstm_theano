import numpy as np
import theano
import theano.tensor as T
from lib import sigmoid, softmax, dropout, floatX, random_weights, zeros


class NNLayer:

    def get_params(self):
        return self.params

    def save_model(self):
        return

    def load_model(self):
        return

    def updates(self):
        return []

    def reset_state(self):
        return

class InputLayer(NNLayer):
    """
    """
    def __init__(self, X, name=""):
        self.name = name
        self.X = X
        self.params=[]

    def output(self, train=False):
        return self.X

class FullyConnectedLayer(NNLayer):
    """
    """
    def __init__(self, num_input, num_output, input_layer, name=""):
        self.num_input = num_input
        self.num_output = num_output
        self.X = input_layer.output()
        self.W_yh = random_weights((num_input, num_output),name="W_yh")
        self.b_y = zeros(num_output, name="b_y")
        self.params = [self.W_yh, self.b_y]

    def output(self):
        return T.dot(self.X, self.W_yh) + self.b_y

    def reset_state(self):
        self.W_yh = random_weights((self.num_input, self.num_output),name="W_yh")
        self.b_y = zeros(self.num_output, name="b_y")
        self.params = [self.W_yh, self.b_y]


class LSTMLayer(NNLayer):

    def __init__(self, num_input, num_cells, input_layer=None, name=""):
        """
        LSTM Layer
        Takes as input sequence of inputs, returns sequence of outputs

        Currently takes only one input layer
        """
        self.name = name
        self.num_input = num_input
        self.num_cells = num_cells

        #Setting the X as the input layer
        self.X = input_layer.output()

        self.h0 = theano.shared(floatX(np.zeros((1, num_cells))))
        self.s0 = theano.shared(floatX(np.zeros((1, num_cells))))


        #Initializing the weights
        self.W_gx = random_weights((num_input, num_cells), name=self.name+"W_gx")
        self.W_ix = random_weights((num_input, num_cells), name=self.name+"W_ix")
        self.W_fx = random_weights((num_input, num_cells), name=self.name+"W_fx")
        self.W_ox = random_weights((num_input, num_cells), name=self.name+"W_ox")

        self.W_gh = random_weights((num_cells, num_cells), name=self.name+"W_gh")
        self.W_ih = random_weights((num_cells, num_cells), name=self.name+"W_ih")
        self.W_fh = random_weights((num_cells, num_cells), name=self.name+"W_fh")
        self.W_oh = random_weights((num_cells, num_cells), name=self.name+"W_oh")

        self.b_g = zeros(num_cells, name=self.name+"b_g")
        self.b_i = zeros(num_cells, name=self.name+"b_i")
        self.b_f = zeros(num_cells, name=self.name+"b_f")
        self.b_o = zeros(num_cells, name=self.name+"b_o")

        self.params = [self.W_gx, self.W_ix, self.W_ox, self.W_fx,
                        self.W_gh, self.W_ih, self.W_oh, self.W_fh,
                        self.b_g, self.b_i, self.b_f, self.b_o,]

        self.output()

    #Function to calculate the next step
    def forward_step(self, x, h_tm1, s_tm1):

        g = T.tanh(T.dot(x, self.W_gx) + T.dot(h_tm1, self.W_gh) + self.b_g)
        i = T.nnet.sigmoid(T.dot(x, self.W_ix) + T.dot(h_tm1, self.W_ih) + self.b_i)
        f = T.nnet.sigmoid(T.dot(x, self.W_fx) + T.dot(h_tm1, self.W_fh) + self.b_f)
        o = T.nnet.sigmoid(T.dot(x, self.W_ox) + T.dot(h_tm1, self.W_oh) + self.b_o)

        s = i*g + s_tm1 * f
        h = T.tanh(s) * o

        return h, s

    def output(self, train=True):

        outputs_info = [self.h0, self.s0]

        ([outputs, states], updates) = theano.scan(
                fn=self.forward_step,
                sequences=self.X,
                outputs_info = outputs_info)

        return outputs

    def reset_state(self):
        self.h0 = theano.shared(floatX(np.zeros(self.num_cells)))
        self.s0 = theano.shared(floatX(np.zeros(self.num_cells)))
