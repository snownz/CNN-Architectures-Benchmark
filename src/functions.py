from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, accuracy_score, classification_report
from torchmetrics.classification import MulticlassF1Score

def train_and_evaluate( model, optimizer, train_loader, val_loader, num_epochs, device, is_sam = False ):
    
    # Move model to the correct device
    model.to( device )

    # Metrics storage
    train_losses, val_losses = [4], [4]
    train_f1_scores, val_f1_scores = [0], [0]

    # Define F1 Score metric
    f1_metric = MulticlassF1Score( num_classes = 10 ).to( device )

    # Initialize Matplotlib figure
    plt.ioff()  # Disable interactive mode for Jupyter Notebook
    fig, axes = plt.subplots( 2, 2, figsize = ( 12, 8 ) )  # 2x2 grid (Train & Val Loss/F1)

    for epoch in range( 1, num_epochs + 1 ):

        model.train()
        running_loss, running_f1, total_samples = 0.0, 0.0, 0

        print( f"Epoch {epoch}/{num_epochs}" )
        
        for batch_idx, ( images, labels ) in enumerate( train_loader ):

            images, labels = images.to( device ), labels.to( device )

            # Forward & backward pass
            if not is_sam:
                
                optimizer.zero_grad()
                loss, logits = model.forward_backward( images, labels )
                optimizer.step()
            
            else:
            
                # First sam step
                loss, logits = model.forward_backward( images, labels )
                optimizer.first_step( zero_grad = True )

                # Second sam step
                _, _ = model.forward_backward( images, labels )
                optimizer.second_step( zero_grad = True )

            # Compute F1 Score
            f1_score = f1_metric( logits.softmax( dim = 1 ), labels )

            # Accumulate loss and F1 Score
            batch_size = labels.size( 0 )
            running_loss += loss.item() * batch_size
            running_f1 += f1_score.item() * batch_size
            total_samples += batch_size

            # Average loss and F1 score
            avg_train_loss = running_loss / total_samples
            avg_train_f1 = running_f1 / total_samples
            
            # Store training metrics
            train_losses.append( 0.9 * train_losses[-1 ] + 0.1 * avg_train_loss )
            train_f1_scores.append( 0.9 * train_f1_scores[-1 ] + 0.1 * avg_train_f1 )

            # Live Update - Train Phase
            if batch_idx % 10 == 0:
                clear_output( wait = True )

                # LOSS PLOTS
                axes[0, 0].clear()
                axes[0, 0].plot( train_losses, label = "Train Loss", marker = 'o', linestyle = '-' )
                axes[0, 0].set_title( "Training Loss Over Time" )
                axes[0, 0].set_xlabel( "Batch" )
                axes[0, 0].set_ylabel( "Loss" )
                axes[0, 0].legend()
                axes[0, 0].grid()

                # F1 SCORE PLOTS
                axes[1, 0].clear()
                axes[1, 0].plot( train_f1_scores, label = "Train F1 Score", marker = 'o', linestyle = '-' )
                axes[1, 0].set_title( "Training F1 Score Over Time" )
                axes[1, 0].set_xlabel( "Batch" )
                axes[1, 0].set_ylabel( "F1 Score" )
                axes[1, 0].legend()
                axes[1, 0].grid()

                display(fig)  # Show updated figure in Jupyter

        # VALIDATION PHASE (Live Update per batch)
        model.eval()
        val_loss, val_f1, total_val_samples = 0.0, 0.0, 0
        
        with torch.no_grad():
            
            for batch_idx, ( images, labels ) in enumerate( val_loader ):

                images, labels = images.to( device ), labels.to( device )

                # Forward pass only
                predictions = model( images )
                loss = F.cross_entropy( predictions, labels )

                # Compute F1 Score
                f1_score = f1_metric( predictions.softmax( dim = 1 ), labels )

                # Accumulate loss and F1 score
                batch_size = labels.size( 0 )
                val_loss += loss.item() * batch_size
                val_f1 += f1_score.item() * batch_size
                total_val_samples += batch_size

                # Average validation loss and F1 score
                avg_val_loss = val_loss / total_val_samples
                avg_val_f1 = val_f1 / total_val_samples

                # Store validation metrics (persists across epochs)
                val_losses.append( 0.9 * val_losses[-1] + 0.1 * avg_val_loss )
                val_f1_scores.append( 0.9 * val_f1_scores[-1] + 0.1 * avg_val_f1 )

                # Live Update - Validation Phase
                clear_output(wait=True)  # Clear previous plot only in Jupyter

                # LOSS PLOTS
                axes[0, 1].clear()
                axes[0, 1].plot( val_losses, label = "Validation Loss", marker = 's', linestyle = '--' )
                axes[0, 1].set_title( "Validation Loss Over Time" )
                axes[0, 1].set_xlabel( "Batch" )
                axes[0, 1].set_ylabel( "Loss" )
                axes[0, 1].legend()
                axes[0, 1].grid()

                # F1 SCORE PLOTS
                axes[1, 1].clear()
                axes[1, 1].plot( val_f1_scores, label = "Validation F1 Score", marker = 's', linestyle = '--' )
                axes[1, 1].set_title( "Validation F1 Score Over Time" )
                axes[1, 1].set_xlabel( "Batch" )
                axes[1, 1].set_ylabel( "F1 Score" )
                axes[1, 1].legend()
                axes[1, 1].grid()

                display(fig)  # Show updated figure in Jupyter

        # Set the plt title to the final metrics
        plt.suptitle( f"Epoch {epoch}/{num_epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Train F1: {avg_train_f1:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f}, Val F1: {avg_val_f1:.4f}" )

    plt.show()

    return train_losses, val_losses, train_f1_scores, val_f1_scores

def evaluation( model, test_loader, device ):

    model.eval()

    with torch.no_grad():
        predictions = []
        targets = []
        for images, labels in test_loader:
            images, labels = images.to( device ), labels.to( device )

            # Forward pass
            logits = model( images )

            # Compute F1 Score, precision and accuracy
            targets.extend( labels.cpu().numpy() )
            predictions.extend( logits.argmax( dim = 1 ).cpu().numpy() )

        f1 = f1_score( targets, predictions, average = 'macro' )
        precision = precision_score( targets, predictions, average = 'macro' )
        accuracy = accuracy_score( targets, predictions )
        report = classification_report( targets, predictions )

    return f1, precision, accuracy, report