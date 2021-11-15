import numpy as np

SupernetCfg = {
    'backbone_cfg': {
        'stem': {
            'type': 'conv',
            'filter': list(np.arange(32, 40 + 1, 4)),
            'depth': [1],
            'kernel': [3],
            'ratio': 1,
            'stride': 2
        },
        'stage_1': {
            'type': 'mbconv1',
            'filter': list(np.arange(16, 24 + 1, 4)),
            'depth': list(np.arange(1, 2 + 1, 1)),
            'kernel': [3],
            'ratio': 1,
            'stride': 1
        },
        'stage_2': {
            'type': 'mbconv6',
            'filter': list(np.arange(24, 32 + 1, 4)),
            'depth': list(np.arange(2, 3 + 1, 1)),
            'kernel': [3],
            'ratio': 6,
            'stride': 2
        },
        'stage_3': {
            'type': 'mbconv6',
            'filter': list(np.arange(40, 48 + 1, 4)),
            'depth': list(np.arange(2, 3 + 1, 1)),
            'kernel': [3, 5, 7, 9],
            'ratio': 6,
            'stride': 2
        },
        'stage_4': {
            'type': 'mbconv6',
            'filter': list(np.arange(80, 88 + 1, 4)),
            'depth': list(np.arange(2, 4 + 1, 1)),
            'kernel': [3, 5, 7, 9],
            'ratio': 6,
            'stride': 2
        },
        'stage_5': {
            'type': 'mbconv6',
            'filter': list(np.arange(112, 128 + 1, 4)),
            'depth': list(np.arange(2, 6 + 1, 1)),
            'kernel': [3, 5, 7, 9],
            'ratio': 6,
            'stride': 1
        },
        'stage_6': {
            'type': 'mbconv6',
            'filter': list(np.arange(192, 216 + 1, 4)),
            'depth': list(np.arange(2, 6 + 1, 1)),
            'kernel': [3, 5, 7, 9],
            'ratio': 6,
            'stride': 2
        },
        'stage_7': {
            'type': 'mbconv6',
            'filter': list(np.arange(320, 352 + 1, 4)),
            'depth': list(np.arange(1, 2 + 1, 1)),
            'kernel': [3, 5, 7, 9],
            'ratio': 6,
            'stride': 1
        },
    },
    'transport_cfg': {
        'level_0': {
            'connection': [0, 1, 2, 3, 4],
            'kernel': [3, 5, 7, 9]
        },  # stage1
        'level_1': {
            'connection': [0, 1, 2, 3, 4],
            'kernel': [3, 5, 7, 9]
        },  # stage2
        'level_2': {
            'connection': [0, 1, 2, 3, 4],
            'kernel': [3, 5, 7, 9]
        },  # stage3
        'level_3': {
            'connection': [0, 1, 2, 3, 4],
            'kernel': [3, 5, 7, 9]
        },  # stage5
        'level_4': {
            'connection': [0, 1, 2, 3, 4],
            'kernel': [3, 5, 7, 9]
        },  # stage7
    },
    'decoder_cfg': {
        'level_0': {
            'connection': [0, 1, 2, 3, 4],
            'kernel': [3, 5, 7, 9]
        },
        'level_1': {
            'connection': [1, 2, 3, 4],
            'kernel': [3, 5, 7, 9]
        },
        'level_2': {
            'connection': [2, 3, 4],
            'kernel': [3, 5, 7, 9]
        },
        'level_3': {
            'connection': [3, 4],
            'kernel': [3, 5, 7, 9]
        },
        'level_4': {
            'connection': [4],
            'kernel': [3, 5, 7, 9]
        }
    }
}
