
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.archs import get_cnn_architecture

class SoftmaxPooling2d(nn.Module):
    
    def __init__(self, kernel_size, stride=None, padding=0, beta=1.0):
     
        super( SoftmaxPooling2d, self ).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.beta = beta

    def forward(self, x):

        # x: (N, C, H, W)
        N, C, H, W = x.shape

        # Extract patches: shape (N, C * (k*k), L), where L is the number of sliding windows.
        patches = F.unfold( x, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding )

        # Compute output dimensions.
        if isinstance(self.kernel_size, int):
            k = self.kernel_size
        else:
            k = self.kernel_size[0]

        H_out = ( H + 2 * self.padding - k ) // self.stride + 1
        W_out = ( W + 2 * self.padding - k ) // self.stride + 1

        # Reshape to (N, C, k*k, H_out, W_out)
        patches = patches.view( N, C, k * k, H_out, W_out )

        # Apply softmax across the pooling region (dim=2)
        weights = F.softmax( self.beta * patches, dim = 2 )
        # Compute the weighted sum (differentiable approximation of max)
        out = ( weights * patches ).sum( dim = 2 )
        return out

class CNNClassifier(nn.Module):

    def __init__(self, architecture_name, intermediate_activation=torch.relu):

        super( CNNClassifier, self ).__init__()

        self.intermediate_activation = intermediate_activation

        self.architecture = get_cnn_architecture( architecture_name )
        self.layers = []
        self.build_layers()

        self.cnn_feature_maps = nn.Sequential( *self.layers )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(        
            nn.Linear( int( self.last_cnn_layer_size ), 512 ),
            intermediate_activation(),
            nn.Linear( 512, 10 )
        )

    def build_layers(self, input_size=(32, 32)):

        self.last_cnn_layer_size = None
        last_out_channels = None
        H, W = input_size  # initial feature map dimensions

        for layer_name, params in self.architecture.items():
            
            if 'layer' in layer_name:
            
                # Add a convolutional layer.
                self.layers.append( nn.Conv2d( **params ) )

                # Update spatial dimensions after convolution.
                K = params.get( 'kernel_size', 1 )
                S = params.get( 'stride', 1 )
                P = params.get( 'padding', 0 )
                H = ( H + 2 * P - K ) // S + 1
                W = ( W + 2 * P - K ) // S + 1

                # Keep track of the output size of the last CNN layer.
                self.last_cnn_layer_size = params['out_channels'] * H * W
                last_out_channels = params['out_channels']

            elif 'bn' in layer_name:
                # Add batch normalization and an activation.
                self.layers.append( nn.BatchNorm2d( **params ) )
                self.layers.append( self.intermediate_activation() )

            elif 'dp' in layer_name:
                # Add a dropout layer.
                self.layers.append( nn.Dropout( **params ) )

            elif 'mp' in layer_name:
                
                # Add a max pooling layer.
                self.layers.append( SoftmaxPooling2d( **params ) )

                # Update spatial dimensions after pooling.
                # Use defaults if not specified.
                K = params.get( 'kernel_size', 2 )
                S = params.get( 'stride', 2 )
                P = params.get( 'padding', 0 )
                H = ( H + 2 * P - K ) // S + 1
                W = ( W + 2 * P - K ) // S + 1

                self.last_cnn_layer_size = last_out_channels * H * W

            else:
                raise ValueError(f"Layer type '{layer_name}' not recognized.")

    def forward(self, x):

        x = self.cnn_feature_maps( x )
        x = self.flatten( x )
        x = self.classifier( x )

        return x
    
    def loss(self, prediction, target):
        return F.cross_entropy( prediction, target )
    
    def forward_backward( self, x, y ):

        predictions = self( x )
        loss = self.loss( predictions, y )
        loss.backward()

        return loss, predictions

class PatchEmbed(nn.Module):

    def __init__(self, in_channels, embed_dim, patch_size):
        
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d( in_channels, embed_dim, kernel_size = patch_size, stride = patch_size )

    def forward(self, x):
        
        # x: [batch, C, H, W]
        x = self.proj( x )  # [batch, embed_dim, H_patch, W_patch]
        # Flatten the patches and transpose: [batch, num_patches, embed_dim]
        x = x.flatten( 2 ).transpose( 1, 2 )
        
        return x

class InceptionBlock(nn.Module):

    def __init__(self, in_channels, out1, out3_reduce, out3, out5_reduce, out5, out_pool):
        
        super(InceptionBlock, self).__init__()
        
        # Branch 1: 1x1 conv.
        self.branch1 = nn.Sequential(
            nn.Conv2d( in_channels, out1, kernel_size = 1 ),
            nn.BatchNorm2d( out1 ),
            nn.ReLU( inplace = True )
        )

        # Branch 2: 1x1 reduction followed by 3x3 conv.
        self.branch2 = nn.Sequential(
            nn.Conv2d( in_channels, out3_reduce, kernel_size = 1 ),
            nn.BatchNorm2d( out3_reduce ),
            nn.ReLU( inplace = True ),
            nn.Conv2d( out3_reduce, out3, kernel_size = 3, padding = 1 ),
            nn.BatchNorm2d( out3 ),
            nn.ReLU( inplace = True )
        )

        # Branch 3: 1x1 reduction followed by 5x5 conv.
        self.branch3 = nn.Sequential(
            nn.Conv2d( in_channels, out5_reduce, kernel_size = 1 ),
            nn.BatchNorm2d( out5_reduce ),
            nn.ReLU( inplace = True ),
            nn.Conv2d( out5_reduce, out5, kernel_size = 5, padding = 2 ),
            nn.BatchNorm2d( out5 ),
            nn.ReLU( inplace = True )
        )

        # Branch 4: 3x3 max pooling followed by 1x1 conv.
        self.branch4 = nn.Sequential(
            nn.MaxPool2d( kernel_size = 3, stride = 1, padding = 1 ),
            nn.Conv2d( in_channels, out_pool, kernel_size = 1 ),
            nn.BatchNorm2d( out_pool ),
            nn.ReLU( inplace = True )
        )
    
    def forward(self, x):

        b1 = self.branch1( x )
        b2 = self.branch2( x )
        b3 = self.branch3( x )
        b4 = self.branch4( x )

        return torch.cat( [ b1, b2, b3, b4 ], dim = 1 )

class EnsembleLoss(nn.Module):

    def __init__(self, alpha=0.5):
    
        super( EnsembleLoss, self ).__init__()
        self.alpha = alpha

    def forward(self, predictions, target, ensemble_weights):
        
        bs, num_tokens, num_classes = predictions.size()

        # --- Token-Level Loss ---
        # Expand target so each token gets the same label.
        token_target = target.unsqueeze(1).expand( bs, num_tokens ).reshape(-1)
        # Reshape predictions so that each token is a separate sample.
        token_logits = predictions.reshape( -1, num_classes )
        token_loss = F.cross_entropy( token_logits, token_target )

        # --- Ensemble Loss ---
        # Average the token logits to get an ensemble prediction per sample.
        ensemble_logits = ( predictions * F.softmax( ensemble_weights[None,:,None], dim = 1 ) ).sum( 1 )
        ensemble_loss = F.cross_entropy( ensemble_logits, target )

        # Combine the losses.
        total_loss = self.alpha * token_loss + ( 1 - self.alpha ) * ensemble_loss

        return total_loss

class PyramidalInceptionClassifier(nn.Module):

    def __init__(self, intermediate_activation=nn.ReLU):

        super(PyramidalInceptionClassifier, self).__init__()

        self.inception_first_stage = nn.Sequential(
            # Downsample from 32x32 to 16x16.
            nn.Conv2d( 3, 64, kernel_size = 3, stride = 2, padding = 1 ),
            nn.BatchNorm2d( 64 ),
            intermediate_activation( inplace = True ),
            # Inception block: the output channels will be the sum of the branches.
            # nn.Dropout( 0.3 ),
            InceptionBlock( in_channels = 64,
                            out1 = 32,
                            out3_reduce = 32, out3 = 32,
                            out5_reduce = 16, out5 = 16,
                            out_pool = 16 )
            # Expected output shape: [bs, 32+32+16+16=96, 16, 16]
        )
        
        self.inception_second_stage = nn.Sequential(
            # Downsample from 16x16 to 8x8.
            # nn.Dropout( 0.3 ),
            nn.Conv2d( 96, 128, kernel_size = 3, stride = 2, padding = 1 ),
            nn.BatchNorm2d( 128 ),
            intermediate_activation( inplace = True ),
            # nn.Dropout( 0.3 ),
            InceptionBlock( in_channels = 128,
                            out1 = 64,
                            out3_reduce = 64, out3 = 64,
                            out5_reduce = 32, out5 = 32,
                            out_pool = 32 )
            # Expected output shape: [bs, 64+64+32+32=192, 8, 8]           
        )

        self.inception_third_stage = nn.Sequential(
            # Downsample from 8x8 to 4x4.
            # nn.Dropout( 0.3 ),
            nn.Conv2d( 192, 256, kernel_size = 3, stride = 2, padding = 1 ),
            nn.BatchNorm2d( 256 ),
            intermediate_activation( inplace = True ),
            # nn.Dropout( 0.3 ),
            InceptionBlock( in_channels = 256,
                            out1 = 128,
                            out3_reduce = 128, out3 = 128,
                            out5_reduce = 64, out5 = 64,
                            out_pool = 64 )
            # Expected output shape: [bs, 128+128+64+64=384, 4, 4]            
        )

        self.downsample = nn.Sequential(
            # Downsample from 4x4 to 2x2.
            # nn.Dropout( 0.3 ),
            nn.Conv2d( 384, 512, kernel_size = 3, stride = 2, padding = 1 ),
            nn.BatchNorm2d( 512 ),
            intermediate_activation( inplace = True ),
            # nn.Dropout( 0.3 ),
            # Expected output shape: [bs, 512, 2, 2]
        )

        # Patch embedding layers for each stage (projecting to 2048-dimensional tokens).
        # For stage 1: output [batch, 96, 16, 16] -> using patch_size = 4 gives 4x4 grid = 16 tokens.
        self.patch_embed1 = PatchEmbed( in_channels = 96, embed_dim = 2048, patch_size = 4 )
        # For stage 2: output [batch, 192, 8, 8] -> using patch_size = 2 gives 4x4 grid = 16 tokens.
        self.patch_embed2 = PatchEmbed( in_channels = 192, embed_dim = 2048, patch_size = 2 )
        # For stage 3: output [batch, 384, 4, 4] -> using patch_size = 2 gives 2x2 grid = 4 tokens.
        self.patch_embed3 = PatchEmbed( in_channels = 384, embed_dim = 2048, patch_size = 2 )
        
        # The prediction module: each token is expected to be a vector of 2048 features.
        self.prediction = nn.Sequential(
            nn.Linear( 2048, 1024 ),
            intermediate_activation(inplace=True),
            nn.Linear( 1024, 10 )
        )

        self.criteria = EnsembleLoss()
        self.flatten = nn.Flatten()
        self.ensemble_weights = nn.Parameter( torch.ones( 37 ), requires_grad = True )
    
    def _forward(self, x):
        
        # Pass through the stages.
        first_stage = self.inception_first_stage( x )               # [bs, 96, 16, 16]
        second_stage = self.inception_second_stage( first_stage )   # [bs, 192, 8, 8]
        third_satge = self.inception_third_stage( second_stage )    # [bs, 384, 4, 4]
        last_stage = self.downsample( third_satge )                 # [bs, 512, 2, 2]

        # --- Patch Embedding (grid splitting) ---
        # For each stage, use the corresponding patch embedding to extract tokens.
        patches1 = self.patch_embed1( first_stage )       # [bs, 16, 2048] from 4x4 grid.
        patches2 = self.patch_embed2( second_stage )      # [bs, 16, 2048] from 4x4 grid.
        patches3 = self.patch_embed3( third_satge )       # [bs, 4, 2048] from 2x2 grid.
        patches4 = self.flatten( last_stage )[:,None]     # [bs, 512*2*2=2048] (flattened)

        # Concatenate all tokens along the token dimension.
        tokens = torch.cat( [ patches1, patches2, patches3, patches4 ], dim = 1 )  # [bs, 16+16+4+1=37, 2048]

        # --- Prediction ---
        # Process each token with the prediction MLP.
        predictions = self.prediction( tokens )  # [bs, 37, 10]

        return predictions
    
    def forward(self, x):

        return ( self._forward( x ) * F.softmax( self.ensemble_weights[None,:,None], dim = 1 ) ).sum( dim = 1 )

    def forward_backward(self, x, y):

        predictions = self._forward( x )
        loss = self.criteria( predictions, y, self.ensemble_weights )
        loss.backward()

        return loss, ( predictions * F.softmax( self.ensemble_weights[None,:,None], dim = 1 ) ).sum( dim = 1 )