from training_pipeline.exp_models.CNN_LSTM_models import *
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import Callback
from keras.models import save_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, errno
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#####################
# Model related utils

def getModel(model_name):
    """Get the correct model for training

    Args:
        model_name (string): name of the model type e.g "FC_LSTM"

    Returns:
        model (object): specified keras model

    """

    model = None
    if model_name == "FC_LSTM":
        model = 0
    elif model_name == "CNN_LSTM":
        model = CNN_LSTM_model()
    elif model_name == "CNN_LSTM_STATEFUL":
        model = 0

    return model

####################
# Data related Utils

def load5hpyData(USE_TITANX=True, data_name='TopAngle100_dataX_dataY.h5'):
    """Load h5py data and return HDF5 object corresponding to X_train, Y_train, X_test, Y_test

        Args:
            USE_TITANX (boolean): set True if using the Linux Computer with TITANX
            data_name (string): name of the dataset e.g 'TopAngle100_dataX_dataY.h5'

        Returns:
            dataX_train (HDF5Matrix object): keras object for loading h5py datasets
            dataY_train (HDF5Matrix object): keras object for loading h5py datasets
            dataX_test (HDF5Matrix object): keras object for loading h5py datasets
            dataY_test (HDF5Matrix object): keras object for loading h5py datasets

        """

    if USE_TITANX:
        data_dir = '/home/zanoi/ZANOI/auditory_hallucinations_data/'
    else:
        data_dir = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/data/'

    data_file = data_dir + data_name  # data_name = 'TopAngle100_dataX_dataY.h5' by default

    # Load first element of data to extract information on video
    with h5py.File(data_file, 'r') as hf:
        print("Reading fukk data from file..")
        dataX_train = hf['dataX_train']  # Adding the [:] actually loads it into memory
        dataY_train = hf['dataY_train']
        dataX_test = hf['dataX_test']
        dataY_test = hf['dataY_test']
        print("dataX_train.shape:", dataX_train.shape)
        print("dataY_train.shape:", dataY_train.shape)
        print("dataX_test.shape:", dataX_test.shape)
        print("dataY_test.shape:", dataY_test.shape)

    # Load data into HDF5Matrix object, which reads the file from disk and does not put it into RAM
    dataX_train = HDF5Matrix(data_file, 'dataX_train')
    dataY_train = HDF5Matrix(data_file, 'dataY_train')
    dataX_test = HDF5Matrix(data_file, 'dataX_test')
    dataY_test = HDF5Matrix(data_file, 'dataY_test')

    return dataX_train, dataY_train, dataX_test, dataY_test

def returnH5PYDatasetDims(USE_TITANX=True, data_name='TopAngle100_dataX_dataY.h5'):
    """Load h5py data and return the dimensions of data in the dataet

            Args:
                USE_TITANX (boolean): set True if using the Linux Computer with TITANX
                data_name (string): name of the dataset e.g 'TopAngle100_dataX_dataY.h5'

            Returns:
                frame_h (int): image height
                frame_w (int): image width
                channels (int): number of channels in image
                audio_vector_dim (int): number of dimensions (or features) in audio vector

            """
    if USE_TITANX:
        data_dir = '/home/zanoi/ZANOI/auditory_hallucinations_data/'
    else:
        data_dir = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/data/'

    data_file = data_dir + data_name  # data_name = 'TopAngle100_dataX_dataY.h5' by default

    with h5py.File(data_file, 'r') as hf:
        print("Reading data sample from file..")
        dataX_sample = hf['dataX_train'][0]  # select one sample from (7233,244,244,3)
        dataY_sample = hf['dataY_train'][0]
        print("dataX_sample.shape:", dataX_sample.shape)
        print("dataY_sample.shape:", dataY_sample.shape)

    (frame_h, frame_w, channels) = dataX_sample.shape  # (224,224,3)
    audio_vector_dim = dataY_sample.shape[0]

    return frame_h, frame_w, channels, audio_vector_dim

########################
# Custom Keras Callbacks

class saveModelOnEpochEnd(Callback):
    """Custom callback for Keras which saves model on epoch end"""
    def on_epoch_end(self, epoch, logs={}):
        # Save the model at every epoch end
        print("Saving trained model...")
        model_prefix = 'CNN_LSTM'
        model_path = "../trained_models/" + model_prefix + ".h5"
        save_model(self.model, model_path,
                   overwrite=True)  # saves weights, network topology and optimizer state (if any)
        return

class LossHistory(Callback):
    """Custom callback for Keras which saves loss history of training and testing data"""
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.test_losses = []

    def on_batch_end(self, batch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.test_losses.append(logs.get('val_loss'))

class AccuracyHistory(Callback):
    """Custom callback for Keras which saves accuracy history of training and testing data"""
    def __init__(self):
        super().__init__()
        self.train_acc = []
        self.test_acc = []

    def on_batch_end(self, batch, logs={}):
        self.train_acc.append(logs.get('acc'))
        self.test_acc.append(logs.get('val_acc'))

##################################
# Plotting and saving figure utils

def plotAndSaveData(param_history,y_label,learning_rate_hp,weight_hp,title="Parameter History"):
    """Saves a matplotlib plot that graphs parameter history

            Args:
                param_history (Callback object): Keras object that contains information on the parameter history
                y_label (string): label for y axis e.g "Loss"
                learning_rate_hp (float): learning rate hyperparameter e.g 4e-7
                weight_hp (float): weight scale hyperparameter e.g 0.005
                title (string): title for saved graph - default="Parameter History"

            Returns:
                None

            """

    param_history = (vars(param_history))

    colors = ['r','b']

    for idx, param in enumerate(param_history):
        param_values = param_history[param]
        epochs = np.arange(len(param_values))  # epochs is 1,2,3...[num items in param values]
        plt.plot(epochs, param_values, colors[idx])

    # generate legend
    red_patch = mpatches.Patch(color='red', label='train')
    blue_patch = mpatches.Patch(color='blue', label='test')
    plt.legend(handles=[blue_patch, red_patch], prop={'size': 10})

    # Plot the graph
    plt.xlabel('epochs')
    plt.ylabel(y_label)
    title = '%s-{lr:%6f}-{ws:%6f}' % (title, learning_rate_hp, weight_hp)
    plt.title(title)
    # plt.show()
    plt.draw()
    file_name = '{lr:%8f}-{ws:%6f}.png' % (learning_rate_hp, weight_hp)
    plt.savefig('../graphs/training_history/' + file_name)
    plt.close()

def makeDir(dir_name):
    """Checks if a directory exists and creates one if it doesn't exist

              Args:
                  dir_name (string): folder name e.g "{lr:0.000597}-{ws:0.000759}"

              Returns:
                  dir_name (string): complete relative directory name e.g "../graphs/predicted_spectrums/{lr:0.000597}-{ws:0.000759}"
    """

    dir_name = '../graphs/predicted_spectrums/' + dir_name
    #print (os.path.exists(dir_name))
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return dir_name

def genAndSavePredSpectrum(model,save_img_path, window_length = 300,USE_TITANX=True, data_name='TopAngle100_dataX_dataY.h5'):
    """Generates and saves predicted spectrums when using the model on an unseen test set

              Args:
                  model (Keras model object): model created during training
                  save_img_path (string): path where to save images e.g "../graphs/predicted_spectrums/{lr:0.000597}-{ws:0.000759}"
                  window_length (int): length of predicted window. Default = 300
                  USE_TITANX (boolean): set True if using the Linux Computer with TITANX
                  data_name (string): name of the dataset e.g 'TopAngle100_dataX_dataY.h5'

              Returns:
                  None
        """
    # Define the external SSD where the dataset resides in
    if USE_TITANX:
        data_dir = '/home/zanoi/ZANOI/auditory_hallucinations_data/'
    else:
        data_dir = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/data/'
    file_name = data_dir + data_name

    # Open the h5py file
    with h5py.File(file_name, 'r') as hf:
        print("Reading data from file..")
        dataX_test = hf['dataX_test'][:]
        dataY_test = hf['dataY_test'][:]
    print("dataX_test.shape:", dataX_test.shape)
    print("dataY_test.shape:", dataY_test.shape)
    print("np.max(dataY)", np.max(dataX_test))
    print("np.min(dataY)", np.min(dataY_test))

    (num_frames, frame_h, frame_w, channels) = dataX_test.shape
    num_windows = math.floor(num_frames / window_length)

    for i in tqdm(range(num_windows)):
        pred_idx = i * window_length
        end_idx = pred_idx + window_length

        trainPredict = model.predict(dataX_test)
        trainScore = math.sqrt(mean_squared_error(dataY_test[pred_idx:end_idx, :], trainPredict[pred_idx:end_idx, :]))
        print('Train score: %.3f RMSE' % (trainScore))

        ##### PLOT RESULTS
        trainPlot = model.predict(dataX_test[pred_idx:end_idx, :])
        print(trainPlot.shape)
        plt.subplot(3, 1, 1)
        plt.imshow(trainPlot.T, aspect='auto')
        plt.title('Model prediction')
        plt.ylabel('Note bins')
        plt.xlabel('Time (frames)')
        plt.subplot(3, 1, 2)
        plt.title('Ground Truth')
        plt.ylabel('Note bins')
        plt.xlabel('Time (frames)')
        plt.annotate('RMSE: %.3f' % (trainScore), xy=(5, 5), xytext=(5, 33))
        plt.imshow(dataY_test[pred_idx:end_idx, :].T, aspect='auto')
        plt.subplot(3, 1, 3)
        plt.imshow(dataY_test[pred_idx:end_idx, :].T, aspect='auto')
        plt.colorbar()
        plt.tight_layout()
        plt.draw()
        plt.savefig(save_img_path + str(i) + '.png')  # ../graphs/predicted_spectrums/{lr:0.000597}-{ws:0.000759}/1.png
        # plt.show()
        plt.close()

def plotAndSaveSession(learning_rates,weight_scales,final_accuracies):
    learning_rates = np.array(learning_rates)
    weight_scales = np.array(weight_scales)
    final_accuracies = np.array(weight_scales)

    print (final_accuracies)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.scatter(learning_rates, weight_scales, final_accuracies)

    ax.set_xlabel('Learning Rates')
    ax.set_ylabel('Weight init')
    ax.set_zlabel('Final accuracy')

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
