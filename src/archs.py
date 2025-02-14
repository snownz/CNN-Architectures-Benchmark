straight_forward_cnn_architectures = {
    
    'simple_cnn': {
        # A basic, straightforward architecture.
        'layer1': {'in_channels': 3, 'out_channels': 16, 'kernel_size': 5, 'stride': 1, 'padding': 2},
        'bn1': {'num_features': 16},
        'dp1': {'p': 0.3},
        'mp1': {'kernel_size': 2, 'stride': 2, 'beta': 5.0},

        'layer2': {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        'bn2': {'num_features': 32},
        'dp2': {'p': 0.3},
        'mp2': {'kernel_size': 2, 'stride': 2, 'beta': 5.0},

        'layer3': {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        'bn3': {'num_features': 64},
        'dp3': {'p': 0.3},
        'mp3': {'kernel_size': 2, 'stride': 2, 'beta': 5.0},

        'layer4': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        'bn4': {'num_features': 128},
        'dp4': {'p': 0.3},
    },

    'deeper_cnn': {
        # A deeper network with more convolutional layers for hierarchical feature extraction.
        'layer1': {'in_channels': 3, 'out_channels': 32, 'kernel_size': 5, 'stride': 1, 'padding': 2},
        'bn1': {'num_features': 32},
        'dp1': {'p': 0.2},
        'mp1': {'kernel_size': 2, 'stride': 2, 'beta': 5.0},

        'layer2': {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        'bn2': {'num_features': 32},
        'dp2': {'p': 0.2},

        'layer3': {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        'bn3': {'num_features': 64},
        'dp3': {'p': 0.3},
        'mp2': {'kernel_size': 2, 'stride': 2, 'beta': 5.0},

        'layer4': {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        'bn4': {'num_features': 64},
        'dp4': {'p': 0.3},

        'layer5': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        'bn5': {'num_features': 128},
        'dp5': {'p': 0.3},
    },

    'wider_cnn': {
        # A network with more filters (wider layers) to capture more feature channels.
        'layer1': {'in_channels': 3, 'out_channels': 64, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        'bn1': {'num_features': 64},
        'dp1': {'p': 0.3},
        'mp1': {'kernel_size': 2, 'stride': 2, 'beta': 5.0},

        'layer2': {'in_channels': 64, 'out_channels': 64, 'kernel_size': 5, 'stride': 1, 'padding': 2},
        'bn2': {'num_features': 64},
        'dp2': {'p': 0.3},

        'layer3': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        'bn3': {'num_features': 128},
        'dp3': {'p': 0.3},
        'mp2': {'kernel_size': 2, 'stride': 2, 'beta': 5.0},

        'layer4': {'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        'bn4': {'num_features': 128},
        'dp4': {'p': 0.3},

        'layer5': {'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        'bn5': {'num_features': 256},
        'dp5': {'p': 0.4},
    },

    'botte_neck_cnn': {
        # A bottleneck design: features are first compressed then expanded,
        # which can reduce computation while preserving important features.
        'layer1': {'in_channels': 3, 'out_channels': 32, 'kernel_size': 5, 'stride': 1, 'padding': 2},
        'bn1': {'num_features': 32},
        'dp1': {'p': 0.2},
        'mp1': {'kernel_size': 2, 'stride': 2, 'beta': 5.0},

        'layer2': {'in_channels': 32, 'out_channels': 16, 'kernel_size': 1, 'stride': 1, 'padding': 0},
        'bn2': {'num_features': 16},

        'layer3': {'in_channels': 16, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        'bn3': {'num_features': 64},
        'dp3': {'p': 0.3},
        'mp2': {'kernel_size': 2, 'stride': 2, 'beta': 5.0},

        'layer4': {'in_channels': 64, 'out_channels': 32, 'kernel_size': 1, 'stride': 1, 'padding': 0},
        'bn4': {'num_features': 32},

        'layer5': {'in_channels': 32, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        'bn5': {'num_features': 128},
        'dp5': {'p': 0.4},
    },

    'compacted_cnn': {
        # A more compact network designed to efficiently capture features with aggressive downsampling.
        'layer1': {'in_channels': 3, 'out_channels': 16, 'kernel_size': 5, 'stride': 1, 'padding': 2},
        'bn1': {'num_features': 16},
        'dp1': {'p': 0.2},
        'mp1': {'kernel_size': 2, 'stride': 2, 'beta': 5.0},

        'layer2': {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        'bn2': {'num_features': 32},
        'dp2': {'p': 0.2},

        'layer3': {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        'bn3': {'num_features': 64},
        'dp3': {'p': 0.3},
        'mp2': {'kernel_size': 2, 'stride': 2, 'beta': 5.0},

        'layer4': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        'bn4': {'num_features': 128},
        'dp4': {'p': 0.3},

        'layer5': {'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        'bn5': {'num_features': 256},
        'dp5': {'p': 0.4},
        'mp3': {'kernel_size': 2, 'stride': 2, 'beta': 5.0},
    }
}

def get_cnn_architecture( architecture_name ):
    if architecture_name in straight_forward_cnn_architectures:
        return straight_forward_cnn_architectures[architecture_name]
    else:
        raise ValueError('Architecture not found')
