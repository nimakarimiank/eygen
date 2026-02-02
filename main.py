
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Average
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy as scc
from keras.datasets import mnist
from keras.regularizers import l2
from tensorflow import config
#import SpectralTools.TensorFlow.spectraltools as spectraltools
import numpy as np
from spectral import Spectral, prune_percentile, metric_based_pruning


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
y = Spectral(200,  **spectral_configuration, name='Spec1')(x)
y = Spectral(300,  **spectral_configuration, name='Spec2')(y)
outputs = Dense(10, activation="softmax", name='LastDense')(y)

model = Model(inputs=inputs, outputs=outputs, name="branched")

compile_dict=dict(optimizer=Adam(1E-3), 
                  loss=scc(from_logits=False), 
                  metrics=["accuracy"])

model.compile(**compile_dict)
model.fit(x_train, y_train, validation_split=0.2, batch_size=300, epochs=1, verbose=1)
model.evaluate(x_test, y_test, batch_size=300)


# Now the 30% of the spectral layers node will be in place pruned according to their relevance. The eigenvalues whose magnitude is smaller than the corresponding percentile will be set to zero by masking the corresponding weights. This will also have an effect on the corresponding bias which will be also masked.
## TODO ValueError: need at least one array to concatenate

pruned_model = prune_percentile(model, 50,
                                compile_dictionary=compile_dict)
print(f'Pruned accuracy: {pruned_model.evaluate(x_test, y_test, batch_size=300)[1]:.3f}')
print('**************************** after Pruned_model ****************************')


print(f'Baseline accuracy: {model.evaluate(x_test, y_test, batch_size=300)[1]:.3f}')
# Cycle through the spectral layers and count the number of active nodes

for lay in pruned_model.layers:
    if hasattr(lay, 'diag_end_mask'):
        print(f'Layer {lay.name} has {np.count_nonzero(lay.diag_end_mask)} active nodes')
    
pruned_model = metric_based_pruning(model, 
                     eval_dictionary=dict(x=x_train, y=y_train, batch_size=200),
                     compile_dictionary=compile_dict,
                     compare_metric='accuracy',
                     max_delta_percent=3)

print(f'Pruned accuracy: {pruned_model.evaluate(x_test, y_test, batch_size=300)[1]:.3f}')
for lay in pruned_model.layers:
    if hasattr(lay, 'diag_end_mask'):
        print(f'Layer {lay.name} has {np.count_nonzero(lay.diag_end_mask)} active nodes')