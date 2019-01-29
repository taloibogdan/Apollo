from .. import components


def generate_architecture(structure):
    filters = structure['filters']
    strides = structure['strides']
    deconv_type = structure['type']

    architecture = []
    for f, s in zip(filters, strides):
        architecture += [{
            'type': deconv_type,
            'params': {
                'filters': f,
                'strides': s
            }
        }]

    return architecture

