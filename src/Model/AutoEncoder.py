from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf


class ClassAutoEncoder:

    def __init__(self, train_data, network_structure, save_folder, model_name="auto_encoder_model.ckpt",
                 contamination=0.01, training_epochs=101, learning_rate=0.001):

        # Parameters
        self.train_data = train_data
        self.network_structure = network_structure
        self.save_folder = save_folder
        self.model_name = save_folder+model_name
        self.contamination = contamination
        self.training_epochs = training_epochs
        self.learning_rate = learning_rate

        # Internal variables
        self.mse = None
        self.train_data_container = None
        self.cost = None
        self.optimizer = None
        self.contamination_threshold = None

    def fit(self):
        """
        Estimate the model
        """
        self.build_auto_encoder()
        self.train_auto_encoder()

    def predict(self, numpy_array):
        """
        Predict if an observation is anomalous or not
        :param numpy_array: Observations to test.
        :return: numpy array of boolean, 1 if anomaly, 0 otherwise
        """
        return self.score_samples(numpy_array) > self.contamination_threshold

    def score_samples(self, numpy_array):
        """
        Compute a score of anomaly for each observation (between 0 and +infinity, the bigger
        :param numpy_array:
        :return anomalies: an array with the quadratics errors between the observations and the results of the
                           auto-encoder
        """
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_name)
            anomalies = sess.run(self.mse, feed_dict={self.train_data_container: numpy_array})
        return anomalies

    def build_auto_encoder(self):
        """
        Build auto-encoder model in TensorFlow
        :return cost: cost function to be optimized
        :return optimizer: optimizer
        """

        # Network Parameters
        n_input = self.train_data.shape[1]  # number of columns of df_anomaly here as our feature dimension

        # Set variable for input data
        self.train_data_container = tf.placeholder("float", [None, n_input])

        # Setting variables for bias and weight
        weights, biases = self.initialize_biases_and_weights()

        # Construct model
        encoder_op = self.build_network(self.train_data_container, weights, biases, "encoder")
        decoder_op = self.build_network(encoder_op, weights, biases, "decoder")

        # Prediction
        y_pred = decoder_op
        # Targets (Labels) are the input data.
        y_true = self.train_data_container

        # Define batch mse
        self.mse = tf.reduce_mean(tf.pow(y_true - y_pred, 2), 1)

        # Contamination threshold
        self.contamination_threshold = tf.contrib.distributions.percentile(self.mse, (1-self.contamination)*100)

        # Define loss and optimizer, minimize the squared error
        self.cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)

    def initialize_biases_and_weights(self):
        """
        Initialize biases and weight
        """
        n_input = self.train_data.shape[1]
        network_structure = self.network_structure
        weights = {}
        biases = {}
        # build encoding layer
        for id_layer, layer in enumerate(network_structure):
            if id_layer == 0:
                weights["encoder_h{}".format(id_layer + 1)] = tf.Variable(
                    tf.random_normal([n_input, network_structure[0]]))
                biases["encoder_b{}".format(id_layer + 1)] = tf.Variable(tf.random_normal([network_structure[0]]))
            else:
                weights["encoder_h{}".format(id_layer + 1)] = tf.Variable(
                    tf.random_normal([network_structure[id_layer - 1],
                                      network_structure[id_layer]]))
                biases["encoder_b{}".format(id_layer + 1)] = tf.Variable(
                    tf.random_normal([network_structure[id_layer]]))

        # build decoding layer
        for id_layer, layer in enumerate(network_structure):
            num_layers = len(network_structure)
            if id_layer == num_layers - 1:
                weights["decoder_h{}".format(id_layer + 1)] = tf.Variable(
                    tf.random_normal([network_structure[0], n_input]))
                biases["decoder_b{}".format(id_layer + 1)] = tf.Variable(tf.random_normal([n_input]))
            else:
                weights["decoder_h{}".format(id_layer + 1)] = tf.Variable(tf.random_normal(
                    [network_structure[num_layers - id_layer - 1], network_structure[num_layers - id_layer - 2]]))
                biases["decoder_b{}".format(id_layer + 1)] = tf.Variable(
                    tf.random_normal([network_structure[num_layers - id_layer - 2]]))

        return weights, biases

    def build_network(self, data, weights, biases, step_type):

        network_structure = self.network_structure

        if step_type == "encoder":
            bias_key = "encoder_b"
            hidden_key = "encoder_h"
        elif step_type == "decoder":
            bias_key = "decoder_b"
            hidden_key = "decoder_h"
        else:
            print("Unknown step type")
            return None

        for id_layer in range(len(network_structure)):
            if id_layer == 0:
                layer = tf.nn.tanh(tf.add(tf.matmul(data, weights["{}{}".format(hidden_key, id_layer + 1)]),
                                          biases["{}{}".format(bias_key, id_layer + 1)]))
            else:
                layer = tf.nn.tanh(tf.add(tf.matmul(prev_layer, weights["{}{}".format(hidden_key, id_layer + 1)]),
                                          biases["{}{}".format(bias_key, id_layer + 1)]))
            prev_layer = layer
        return layer

    def train_auto_encoder(self):
        """
        Train auto-encoder
        """

        # Parameters
        batch_size = 256
        display_step = 1

        # define where model will be saved
        saver = tf.train.Saver()

        # Initializing the variables
        init = tf.global_variables_initializer()

        total_error_sequence = []
        batch_error_sequence = []

        with tf.Session() as sess:
            now = datetime.now()
            sess.run(init)
            total_batch = int(self.train_data.shape[0] / batch_size)
            # Training cycle
            for epoch in range(self.training_epochs):
                # Loop over all batches
                for i in range(total_batch):
                    batch_idx = np.random.choice(self.train_data.shape[0], batch_size)
                    batch_xs = self.train_data[batch_idx]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([self.optimizer, self.cost], feed_dict={self.train_data_container: batch_xs})

                # Display logs per epoch step
                if epoch % display_step == 0:
                    totalcost = sess.run(self.cost, feed_dict={self.train_data_container: self.train_data})
                    print("Epoch:", '%04d' % (epoch + 1),
                          "cost by batch=", "{:.9f}".format(c),
                          "Total cost {:.9f}".format(totalcost),
                          "Time elapsed=", "{}".format(datetime.now() - now))
                    batch_error_sequence.append(c)
                    total_error_sequence.append(totalcost)

            self.contamination_threshold = sess.run(self.contamination_threshold,
                                                    feed_dict={self.train_data_container: self.train_data})
            print("contamination threshold:", self.contamination_threshold)
            print("Optimization Finished!")

            saver.save(sess, self.model_name)
            print("Model saved in file: {}".format(self.model_name))

        error_series = pd.Series(batch_error_sequence)
        error_series.plot()


