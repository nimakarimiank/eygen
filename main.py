
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Average
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy as scc
from keras.datasets import mnist
from keras.regularizers import l2
from tensorflow import config
import numpy as np
from spectral import Spectral, prune_percentile, metric_based_pruning
import argparse as ap
parser = ap.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=300, help='Batch size for training and evaluation')
parser.add_argument('--prune_percentile', type=float, default=90, help='Percentile for pruning the model')
parser.add_argument('--layer_nodes', type=int, nargs='+', default=[20, 30], help='Number of nodes in each spectral layer')
args = parser.parse_args()
physical_devices = config.experimental.list_physical_devices('GPU')
for dev in physical_devices:
    config.experimental.set_memory_growth(dev, True)








# Dataset and model creation
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.
x_test = x_test / 255.

spectral_configuration = {'activation': 'relu', 
                          'use_bias': True,
                          'base_regularizer': l2(1E-3),
                          'diag_regularizer': l2(5E-3)}

inputs = Input(shape=(28, 28,))
x = Flatten()(inputs)
for i, nodes in enumerate(args.layer_nodes):
    x = Spectral(nodes, **spectral_configuration, name=f'Spectral_{i}')(x)
outputs = Dense(10, activation="softmax", name='LastDense')(x)

model = Model(inputs=inputs, outputs=outputs, name="branched")

compile_dict=dict(optimizer=Adam(1E-3), 
                  loss=scc(from_logits=False), 
                  metrics=["accuracy"])

model.compile(**compile_dict)
model.fit(x_train, y_train, validation_split=0.2, batch_size=args.batch_size, epochs=args.epochs, verbose=1)
model.evaluate(x_test, y_test, batch_size=args.batch_size)

print('**************************** before Pruned_model ****************************')

print(f'Baseline accuracy: {model.evaluate(x_test, y_test, batch_size=args.batch_size)[1]:.3f}')
print('*********************************************************************************')

print('**************************** after Pruned_model ****************************')
pruned_model = prune_percentile(model, args.prune_percentile,
                                compile_dictionary=compile_dict)
print(f'Pruned accuracy: {pruned_model.evaluate(x_test, y_test, batch_size=args.batch_size)[1]:.3f}')
print('*********************************************************************************')




# # Cycle through the spectral layers and count the number of active nodes

# for lay in pruned_model.layers:
#     if hasattr(lay, 'diag_end_mask'):
#         print(f'Layer {lay.name} has {np.count_nonzero(lay.diag_end_mask)} active nodes')
    
# pruned_model = metric_based_pruning(model, 
#                      eval_dictionary=dict(x=x_train, y=y_train, batch_size=200),
#                      compile_dictionary=compile_dict,
#                      compare_metric='accuracy',
#                      max_delta_percent=3)

# print(f'Pruned accuracy: {pruned_model.evaluate(x_test, y_test, batch_size=300)[1]:.3f}')
# for lay in pruned_model.layers:
#     if hasattr(lay, 'diag_end_mask'):
#         print(f'Layer {lay.name} has {np.count_nonzero(lay.diag_end_mask)} active nodes')