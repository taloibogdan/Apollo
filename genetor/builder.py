import tensorflow as tf
from . import architectures
# import .components
from . import components
from .components.default_params import DEFAULT_PARAMS
from .architectures.default_structures import DEFAULT_STRUCTURE


def new_graph(architecture = [], input = None):

    last_layer = input
    node_count = {}

    for node in architecture:
        node_count[node['type']] = node_count.get(node['type'], -1) + 1
        node_name = '{}_{}'.format(node['type'],
                                   node_count[node['type']])

        if 'input' in node:
            if type(node['input']) is str:
                last_layer = tf.get_default_graph().get_tensor_by_name(node['input'])
            else:
                last_layer = node['input']

        node_params = node.get('params', {})
        params = dict(DEFAULT_PARAMS.get(node['type'], {}),
                      **node_params,
                      name = node_name)

        layer = getattr(components, node['type'])(last_layer, **params)

        if 'output_label' in node:
            layer = tf.identity(layer,
                                name = node['output_label'])

        last_layer = layer

    return last_layer


def new_architecture(model, structure):
    model_generator = getattr(architectures, model)
    
    structure = dict(DEFAULT_STRUCTURE.get(model, {}),
                     **structure)

    return model_generator.generate_architecture(structure)
    



