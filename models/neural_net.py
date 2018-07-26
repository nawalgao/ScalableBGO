import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
import sys
sys.path.append('../')
from misc import normalization


class NeuralNet(object):
    def __init__(self, X, y, rng, normalize_input, normalize_output):
        """
        Parameters:
        ------------
        X : np.ndarray(N, D)
        Input data points. Dimensionality of X is (N , D),
        with N as the number of datapoints and D is the number of features
        y : np.ndarray(N,)
        Corresponding target values
        """
        self.X = X
        self.num_feat = X.shape[1]
        self.y = y
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0,1000))
        else:
            self.rng = rng

    def build_nn(self, n_units_1, n_units_2, n_units_3):
        """
        Build a Keras neural network representation (build function) 
        which would then be used by KerasRegressor to further train the layer
        """
        
        # create model
        model = Sequential()
        model.add(Dense(n_units_1, input_dim = self.num_feat,
                                     kernel_initializer='normal',
                                     activation='tanh'))
        model.add(Dense(n_units_2, kernel_initializer='normal',
                                     activation='tanh'))
        model.add(Dense(n_units_3, kernel_initializer = 'normal',
                                     activation = 'tanh', name = 'll_out'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        
        return model

    def train_nn(self, num_epochs, batch_size,
                 n_units_1, n_units_2, n_units_3):
        """
        Trains the NN part of our model

        Parameters:
        ----------
        batch_size : int
            Batch size for training the neural network
        num_epochs : int
            Number of epochs for training
        n_units_1 : int
            Number of units in layer 1
        n_units_2 : int
            Number of units in layer 2
        n_units_3 : int
            Number of units in layer 3
        """
        if self.normalize_input:
            self.X, self.xnorm_mean, self.xnorm_sd  = normalization.zero_mean_unit_var_normalization(self.X)
        if self.normalize_output:
            self.y, self.ynorm_mean, self.ynorm_sd = normalization.zero_mean_unit_var_normalization(self.y)
        nn_model = self.build_nn(n_units_1, 
                                 n_units_2, n_units_3)
        nn_model.fit(self.X, self.y, batch_size = batch_size,
                     nb_epoch = num_epochs, verbose=1)
        self.nn_model = nn_model

        return nn_model 
    
    def extract_last_layer_nn_out(self, X_test):
        """
        Extract the output of the last layer of the trained neural network
        Outputs:
            ll_out : Output of the last layer of the neural network
        """
        if self.normalize_input:
            X_test = normalization.zero_mean_unit_var_normalization(X_test)[0]
            
        layer_name = 'll_out'
        intermediate_layer_model = Model(inputs = self.nn_model.input,
                                         outputs = self.nn_model.get_layer(layer_name).output)
        ll_out = intermediate_layer_model.predict(X_test)

        return ll_out
    
    def predict(self, X_test):
        """
        Using NN, predict the objective function values for testing set
        Inputs:
            X_test : np.ndarray(Nt, D)
        Input data points. Dimensionality of X is (Nt , D),
        with Nt as the number of testing datapoints and D is the number of features
        """
        
        if self.normalize_input:
            X_test = normalization.zero_mean_unit_var_normalization(X_test)[0]
        pred = self.nn_model.predict(X_test)
        pred = normalization.zero_mean_unit_var_unnormalization(pred,
                                                                self.ynorm_mean,
                                                                self.ynorm_sd )
        
        return pred
        
        

if __name__ == '__main__':
   
    import matplotlib.pyplot as plt

    X_train = np.linspace(0, 80, 100).reshape(-1, 1)
    Y_train = 5 * X_train
    
    #plt.scatter(X_train, Y_train)
    #plt.show()

    NN = NeuralNet(X_train, Y_train, rng = 1,
                       normalize_input = True,
                       normalize_output = False)
    tt = NN.train_nn(50000, 500, 50, 50, 50)
    
    ll_out = NN.extract_last_layer_nn_out(X)
    
    X_norm = normalization.zero_mean_unit_var_normalization(X_train)[0]

    pred = tt.predict(X_norm)
