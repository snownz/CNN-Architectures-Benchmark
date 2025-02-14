# Comparative Analysis of CNN Architectures

This report details a comparison between several Convolutional Neural Network (CNN) architectures and a custom **Pyramidal Inception** model. We will discuss each model's design, examine performance metrics (loss, F1-score, accuracy, and precision), and draw conclusions about their relative effectiveness.

---

## 1. Architectures Overview

Below is a brief description of each architecture tested:

1. **simple_cnn**  
   - A shallow, straightforward CNN.
   - Features:
     - 4 convolutional layers of increasing channel depth: \[16, 32, 64, 128\]
     - Dropout (`p=0.3`) in each layer
     - Max-Pooling after layers 1, 2, and 3 with kernel size 2 and stride 2
   - **Goal**: Provide a baseline.

2. **deeper_cnn**  
   - Focuses on *depth* (more layers).
   - Features:
     - Convolutional layers: \[32, 32, 64, 64, 128\] output channels
     - Multiple convolutional blocks before each Max-Pool
     - Dropouts range from `p=0.2` to `p=0.3`
   - **Goal**: Capture hierarchical feature representation by stacking more layers.

3. **wider_cnn**  
   - Prioritizes *width* (i.e., higher number of filters).
   - Features:
     - Convolutional layers: \[64, 64, 128, 128, 256\] output channels
     - Dropouts (`p=0.3` to `0.4`) 
     - Uses kernel sizes 7 and 5 in initial layers to capture wider spatial context
     - Max-Pooling after two specific blocks
   - **Goal**: Increase representational capacity by having more filters at each stage.

4. **botte_neck_cnn**  
   - Incorporates a **bottleneck design**: compress channels first, then expand.
   - Features:
     - Sequence of \[32 → 16 → 64 → 32 → 128\] across the layers, using 1×1 convolutions for compression
     - Dropouts around `p=0.2–0.3`
     - Two Max-Pool operations
   - **Goal**: Achieve an efficient architecture (fewer parameters in the 'bottleneck' layers) while retaining representational power.

5. **compacted_cnn**  
   - A more **compact** architecture with aggressive downsampling.
   - Features:
     - Convolutional layers: \[16, 32, 64, 128, 256\]
     - Several dropout layers (\(p=0.2–0.4\))
     - Max-Pooling in three places
   - **Goal**: Compress spatial dimensions quickly to reduce computational burden.

6. **pyramidal_inception**  
   - Custom multi-stage model with **Inception blocks** and **patch embeddings**:
     - Three inception stages:
       1. From 64 channels, splits into multiple branches: `1x1`, `3x3`, `5x5`, and `pool` branch.
       2. Repeats with 128 channels.
       3. Repeats with 256 channels.
     - Each stage is followed by a downsampling layer (stride=2) to reduce spatial size.
     - Produces multi-scale feature maps of shape: 
       - Stage 1: \([bs, 96, 16,16]\)
       - Stage 2: \([bs,192, 8, 8]\)
       - Stage 3: \([bs,384, 4, 4]\)
       - Downsampled: \([bs,512, 2, 2]\)
     - **Patch Embeddings** at each stage to create a set of tokens (i.e., flatten patches into vectors).
       - For example, in Stage 1: patch size 4 → forms 16 patches each with 2048 features.
     - **Ensemble** of tokens: each token is processed by a linear prediction head, combined with learnable weights to make a single final prediction.
   - **Goal**: Leverage multi-scale processing (Inception) + token-based ensembling for robust classification.

---

## 2. Performance Metrics

The table below summarizes the key metrics recorded after training:

| **Model**            | **Train Loss** | **Val Loss** | **Train F1** | **Val F1**  | **Val Precision** | **Val Accuracy** |
|:---------------------|:--------------:|:------------:|:------------:|:-----------:|:-----------------:|:----------------:|
| **simple_cnn**       | 0.5440         | 1.4802       | 0.7943       | 0.5045      | 0.6943            | 0.5349           |
| **deeper_cnn**       | 0.5351         | 0.7850       | 0.7965       | 0.7070      | 0.7634            | 0.7307           |
| **wider_cnn**        | 0.4465         | 0.7037       | 0.8315       | 0.7429      | 0.7805            | 0.7591           |
| **botte_neck_cnn**   | 0.4389         | 0.7493       | 0.8325       | 0.7243      | 0.7543            | 0.7441           |
| **compacted_cnn**    | 0.5584         | 0.8814       | 0.7924       | 0.6978      | 0.7573            | 0.7187           |
| **pyramidal_inception** | 0.5909     | 0.6201       | 0.9723       | 0.8080      | 0.8241            | 0.8234           |

---

## 3. Analysis

1. **Overall Accuracy & F1**
   - The **pyramidal_inception** approach yields the **highest validation accuracy (82.34%)** and the highest validation F1-score (80.80%). 
     - Notably, it achieves a very high train F1 (97.23%) while still maintaining strong generalization.
   - In terms of the other architectures:
     - **wider_cnn** (75.91% accuracy, 74.29% F1) stands out among the “standard” CNNs.
     - **deeper_cnn** and **botte_neck_cnn** follow closely but remain ~7–8% behind in validation accuracy relative to pyramidal_inception.
     - **simple_cnn** struggles the most on validation performance (53.49% val accuracy), indicating that the shallow design may be insufficiently expressive for this task.
     - **compacted_cnn** also lags somewhat, likely because of very aggressive downsampling that could remove too much spatial information early in the pipeline.

2. **Train vs Validation Loss**
   - The largest gap in train vs validation performance is with **pyramidal_inception**, but ironically, it maintains the best validation metrics. Its train loss is not the absolute lowest (0.5909) compared to, for example, `botte_neck_cnn` (0.4389) or `wider_cnn` (0.4465), yet it generalizes better.
   - This suggests that **pyramidal_inception** has learned a robust representation (through inception modules and ensemble weighting) that translates well to the validation set.

3. **Model Complexity and Feature Extraction**
   - **wider_cnn** shows that increasing the channel width helps capture more complex features, improving the overall accuracy and F1 over the simpler designs.
   - **deeper_cnn** also provides improved accuracy compared to a shallow CNN, reinforcing the general benefit of deeper networks in extracting hierarchical features.
   - **botte_neck_cnn** leverages 1×1 compressions to reduce computational overhead but still obtains a strong result. This aligns with the well-known “bottleneck” concept from ResNet and other modern architectures.
   - **pyramidal_inception** leverages multi-scale feature extraction (Inception blocks) in a pyramidal fashion (repeated stages of increasing channel depth) plus a novel token-based ensemble approach. These design choices appear to be especially effective for capturing robust features at multiple scales.

4. **Ensemble Token Approach**
   - The **pyramidal_inception** architecture uses **PatchEmbed** modules and concatenates token representations from each stage, culminating in an ensemble of 37 tokens (16 + 16 + 4 + 1). Each token is passed through the same classification head, and the results are combined via a learnable weight vector.
   - This method likely aids in capturing different levels of granular information, from early features (large spatial dimension, fewer channels) to late features (small spatial dimension, richer channels). 

---

## 4. Conclusions

1. **Shallow or Aggressive Downsampling** (simple_cnn and compacted_cnn) tends to yield poorer results, suggesting that either insufficient depth or too rapid a loss of spatial resolution reduces feature quality.
2. **Increasing Width or Depth** individually improves performance (wider_cnn and deeper_cnn).
3. **Bottleneck Mechanisms** are effective at balancing efficiency with representational power (botte_neck_cnn).
4. **Multi-Stage Inception + Patch Embedding + Token Ensemble** (pyramidal_inception) provides the strongest result. The combination of:
   - Multi-scale inception blocks,
   - Repeated downsampling stages,
   - A token-based approach with learnable ensemble weights,

   leads to **the highest accuracy (82.34%) and highest F1-score (80.80%)** on the validation set.

In summary, for this particular problem, the **pyramidal_inception** approach significantly outperforms the standard CNN variants—showcasing the value of **multi-scale feature extraction** and **ensemble-based token aggregation**.

---

### Acknowledgments

All training and validation experiments were run under comparable conditions, ensuring consistency across architectures. The results highlight how architectural innovations (Inception blocks, token ensembles) can yield notable gains over more traditional, straightforward CNN designs.


# Ensemble Weight Distribution Analysis

From the **"Ensemble Weights Distribution"**, we have a total of 37 ensemble weights (indexed roughly from 0 to 36). Most of these weights are small—typically below 1 or 2—while one of them (the final token index) is noticeably **much** higher (over 40). 

## Key Observations

1. **Dominance of the Last Token**
   - In the **Pyramidal Inception** architecture, the last token corresponds to the flattened output from the final convolutional/downsampling stage (`x4`), which is a \([bs, 1, 2048]\) representation.  
     - This is a highly downsampled but richly channeled feature map (512 channels, 2×2 spatial dimension).

2. **Comparatively Uniform but Small Weights for Earlier Tokens**
   - The weights for the other tokens (indices 0 through ~35) are relatively small, hovering close to the 1.0 or sub-1.0 region, which indicates they still contribute some information but less dominantly compared to the final stage’s token.

3. **My Interpretation**
   - The large weight on the last token implies that **the final stage of the network** (where features are high-level and abstract) captures the most discriminative information for classification.
   - The earlier-stage tokens (Inception outputs from 16×16, 8×8, and 4×4 resolutions) may provide **complementary** or **fine-grained** features. However, once the model can rely on the more “global” final token, it chooses to do so.
   - In ensemble terms, this is reminiscent of how a strong “expert” can overshadow multiple weaker experts, even if those weaker experts do occasionally contribute beneficial nuances.

## Conclusion

The chart shows a clear preference for the **final token** representation in the Pyramidal Inception model’s ensemble. 
While the network still gives some (minimal) weight to tokens from earlier stages, the overwhelmingly large weight on the last token indicates the model relies heavily on high-level features extracted at the end of the pipeline for making predictions.

---

# Head-Level Predictions Across Model Stages

These plots show **per-token (head) predictions** from the Pyramidal Inception model on various CIFAR-10-like images (e.g. car, dog, ship, bird, horse, deer, etc.). Each horizontal axis is split by vertical dashed lines, corresponding to the boundaries between stages:

- **Stage 1** tokens (indices 0 to 15)
- **Stage 2** tokens (indices 16 to 31)
- **Stage 3** tokens (indices 32 to 35)
- **Stage 4** token (index 36)

Remember that the model produces a separate prediction for each “token” (i.e., each patch-embedded feature map from a given stage), and then **learns a weighted ensemble** of all these predictions.

## 1. Early Tokens Show Greater Variation
Looking at tokens from **Stage 1** and **Stage 2**, the predicted classes can jump around among multiple possibilities (e.g., car → truck → dog or plane → ship → etc.). This reflects:
- **Local or partial features** driving classification in these early layers.
- Less refinement of the representation, so each token can momentarily latch onto sub-features that resemble other classes.

## 2. Later Tokens Become More Consistent
By **Stage 3** (tokens at indices 32 to 35) and **Stage 4** (token at index 36), the predictions stabilize. In many cases, you see the final token (index 36) picking the (correct) class with high confidence:
- This final token corresponds to the **deepest** (most-downsampled, highest-channel) feature representation.
- It tends to capture **global** image information, making it more discriminative.

## 3. One Token to Rule Them All
The **last token** (index 36) very often provides the “best guess.” We also know from the **ensemble weight distribution** that the final token’s weight is large compared to others—meaning, in the ensemble, it tends to dominate. 

## 4. Interpretation & Implications
1. **Refinement Over Stages** – Early-stage tokens (first 16 from Stage 1, next 16 from Stage 2) can be thought of as partial or more localized guesses. As we progress, the model aggregates them but heavily relies on the deeper stage(s), which produce more accurate predictions.
2. **Multi-Scale Insight** – Even though the final token is dominant, the earlier tokens still provide complementary signals that can help refine the final result (especially for tricky or ambiguous images).

Overall, these per-token predictions show the **progressive honing** of classification from **local/partial** to **global/refined** features, culminating in a final token that nearly always aligns with the correct label.

<br>
<br>
<br>
<br>

## Results

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
  <img src="images/p1.png" style="width: 100%;">
  <img src="images/p2.png" style="width: 100%;">
  <img src="images/p3.png" style="width: 100%;">
  <img src="images/p4.png" style="width: 100%;">
  <img src="images/p5.png" style="width: 100%;">
  <img src="images/p6.png" style="width: 100%;">
  <img src="images/p7.png" style="width: 100%;">
  <img src="images/p8.png" style="width: 100%;">
  <img src="images/p9.png" style="width: 100%;">
  <img src="images/p10.png" style="width: 100%;">
  <img src="images/p11.png" style="width: 100%;">
  <img src="images/p12.png" style="width: 100%;">
  <img src="images/p13.png" style="width: 100%;">
  <img src="images/p14.png" style="width: 100%;">
  <img src="images/p15.png" style="width: 100%;">
</div>

---

## References
- Szegedy et al., "Going Deeper with Convolutions" (Inception Networks)
- He et al., "Deep Residual Learning for Image Recognition" (ResNet)
- Dosovitskiy et al., "An Image is Worth 16x16 Words" (Vision Transformers)

---