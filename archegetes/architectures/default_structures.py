DEFAULT_STRUCTURE = {
    'resnet': {
        'filters': [64, 128, 256, 256, 512, 512, 512, 512],
        'headless': True
    },
    'unet': {
        'filters': [32, 64],
        'scope': 'unet'
    }
}

