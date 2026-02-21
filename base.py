## python3 main.py --epochs 20 --batch_size 300 --prune_percentile 90 --layer_nodes 200 300
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Average
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy as scc
from keras.datasets import mnist
from keras.regularizers import l2
from tensorflow import config
import numpy as np
from spectral import Spectral, prune_percentile, metric_based_pruning
from logger import log_run, format_nodes
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=300, help='Batch size for training and evaluation')
parser.add_argument('--prune_percentile', type=float, default=90, help='Percentile for pruning the model')
parser.add_argument('--layer_nodes', type=int, nargs='+', default=[20, 30], help='Number of nodes in each spectral layer')
args = parser.parse_args()
layer_summary = format_nodes(args.layer_nodes + [10])
layer_count = len(args.layer_nodes) + 1
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
    nodes = int(nodes * (100 - args.prune_percentile) / 100)
    print(f'nodes: {nodes}')
    x = Spectral(nodes, **spectral_configuration, name=f'Spectral_{i}')(x)
outputs = Dense(10, activation="softmax", name='LastDense')(x)

model = Model(inputs=inputs, outputs=outputs, name="branched")

compile_dict=dict(optimizer=Adam(1E-3), 
                  loss=scc(from_logits=False), 
                  metrics=["accuracy"])

model.compile(**compile_dict)
model.fit(x_train, y_train, validation_split=0.2, batch_size=args.batch_size, epochs=args.epochs, verbose=1)
baseline_train_loss, baseline_train_acc = model.evaluate(
    x_train, y_train, batch_size=args.batch_size, verbose=0
)
baseline_test_loss, baseline_test_acc = model.evaluate(
    x_test, y_test, batch_size=args.batch_size, verbose=0
)

print('**************************** before Pruned_model ****************************')

print(f'Baseline accuracy: {baseline_test_acc:.3f}')
log_run(
    is_pruned=False,
    no_layers=layer_count,
    no_nodes=layer_summary,
    no_epochs=args.epochs,
    no_batches=args.batch_size,
    pruned_percentile=0.0,
    train_accuracy=baseline_train_acc,
    test_accuracy=baseline_test_acc,
)
print('*********************************************************************************')
