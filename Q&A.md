# DeepShield AI Viva Q&A (Training + Deployment)

This file contains common viva/interview questions with detailed but simple answers and real-time examples based on your DeepShield project.

---

## 1) What problem does your model solve?
**Answer:**  
Our model performs binary classification for deepfake detection: it predicts whether an image/video frame is `authentic` or `deepfake`.

**Real-time example:**  
A journalist receives a suspicious face image on social media. Our model returns `deepfake` with confidence (for example, 0.91), helping quick verification.

---

## 2) Why did you choose deep learning instead of traditional ML?
**Answer:**  
Deep learning (CNNs) learns image features automatically (texture artifacts, blending inconsistencies, face region cues). Traditional ML needs manual feature engineering and usually performs worse on raw image pixels.

**Real-time example:**  
With ML, we must manually design features like noise statistics. With CNN, the network learns these patterns directly from training data.

---

## 3) Which backbones did you compare and why?
**Answer:**  
We compared `MobileNetV2`, `EfficientNetB0`, `ResNet50`, and `Xception` to balance speed vs accuracy.
- MobileNetV2: fast and lightweight
- EfficientNetB0: strong accuracy-efficiency tradeoff
- ResNet50: stable and widely used
- Xception: often strong for deepfake benchmark tasks

---

## 4) Why is data splitting important?
**Answer:**  
If train and test are not separated correctly, model performance looks artificially high. Proper split ensures generalization.

**Real-time example:**  
If frames from the same video appear in both train and test, the model can memorize identity/background and falsely report near-100% accuracy.

---

## 5) What is data leakage?
**Answer:**  
Data leakage happens when test information indirectly appears during training, causing unrealistic metrics.

**Real-time example:**  
Frame-level random split from the same source video is leakage for deepfake detection.

---

## 6) How did you reduce leakage in this project?
**Answer:**  
We introduced **group-based split** (`--split-strategy group`) so related images stay in one split only (train OR val OR test), not multiple.

---

## 7) Why do we use train/val/test separately?
**Answer:**  
- Train: learn parameters  
- Validation: tune hyperparameters and threshold  
- Test: final unbiased performance estimate

---

## 8) Why can 100% accuracy be suspicious?
**Answer:**  
It usually suggests leakage, easy test set, duplicate samples, or overfitting.

**Real-time example:**  
Model gives 100% on internal test but fails on user-uploaded internet images.

---

## 9) What metrics do you track besides accuracy?
**Answer:**  
`Precision`, `Recall`, `F1`, `ROC-AUC`, `Specificity`, `Balanced Accuracy`, and `Confusion Matrix`.

---

## 10) Why is F1 score important?
**Answer:**  
F1 balances Precision and Recall, useful when class distribution is imbalanced.

**Real-time example:**  
If fake samples are fewer, accuracy may look good but model may still miss many deepfakes; F1 exposes that.

---

## 11) What is ROC-AUC in simple terms?
**Answer:**  
ROC-AUC measures ranking quality across all thresholds. Closer to 1 means better separation between classes.

---

## 12) What is threshold tuning?
**Answer:**  
Model outputs probability. Threshold converts probability to class label. Default is 0.5, but tuned threshold can improve business metric.

**Real-time example:**  
Using threshold 0.57 instead of 0.5 reduces false alarms while keeping deepfake detection strong.

---

## 13) Why do you use augmentation?
**Answer:**  
To improve robustness and reduce overfitting. Examples: random flip, zoom, rotation, contrast.

---

## 14) What is class imbalance and how do you handle it?
**Answer:**  
Class imbalance means one class has much more samples than the other. We handle it using balanced sampling and class weights.

---

## 15) What optimizer and loss are used?
**Answer:**  
`Adam` optimizer with `BinaryCrossentropy` loss (with optional label smoothing).

---

## 16) What is transfer learning?
**Answer:**  
Use pretrained backbone weights (ImageNet), then fine-tune on deepfake dataset. This trains faster and needs less data than training from scratch.

---

## 17) What is fine-tuning?
**Answer:**  
After warm-up training with frozen backbone, we unfreeze top layers and continue training at lower learning rate.

---

## 18) Why monitor `val_auc` or `val_accuracy`?
**Answer:**  
Model checkpoint saves best-performing epoch on validation metric, preventing poor final epoch selection.

---

## 19) What is overfitting and how do you detect it?
**Answer:**  
Overfitting means model memorizes training data and performs poorly on unseen data.

**Signs:**  
Train accuracy high, validation/test metrics significantly lower.

**Controls:**  
augmentation, dropout, early stopping, better split strategy, more diverse data.

---

## 20) What is underfitting?
**Answer:**  
Model is too simple or not trained enough. Both train and val performance remain low.

---

## 21) Why do results differ on uploaded user data?
**Answer:**  
Domain shift: training data and real-world data differ in compression, camera quality, ethnicity distribution, lighting, editing style.

---

## 22) How do you improve generalization to real-world uploads?
**Answer:**  
- Use diverse data sources  
- Add compression/noise/blur augmentations  
- Use strict leakage-free split  
- Evaluate on external unseen dataset

---

## 23) How do you evaluate video deepfakes in your app?
**Answer:**  
We sample frames from video, run image-level model on frames, then aggregate frame predictions.

---

## 24) Why can frame-sampling miss manipulations?
**Answer:**  
Temporal artifacts may appear only in motion patterns. Frame-only approach may miss those.

---

## 25) What is the difference between image model and video model?
**Answer:**  
Image model uses spatial features per frame.  
Video models (3D CNN/transformer) use spatial + temporal signals.

---

## 26) Why is reproducibility important?
**Answer:**  
If runs are not reproducible, results cannot be trusted in research or production.

**How:** fixed seeds, controlled data split, versioned dependencies.

---

## 27) Why did backend throw model loading errors?
**Answer:**  
Model file was saved with different Keras/TensorFlow config keys than current runtime supports (example: legacy `BatchNormalization` fields).

---

## 28) What are common deployment risks in ML projects?
**Answer:**  
- Dependency version mismatch  
- Wrong preprocessing at inference  
- Wrong threshold in production  
- Missing model file/paths  
- Data drift over time

---

## 29) What is preprocessing mismatch?
**Answer:**  
If training preprocess differs from inference preprocess, performance drops.

**Real-time example:**  
Applying `preprocess_input` twice can significantly degrade predictions.

---

## 30) How do you choose best model among multiple backbones?
**Answer:**  
Use validation/test metrics with clear priority (for example accuracy-first or F1-first), then choose based on target use case (safety vs speed).

---

## 31) Why not use only accuracy as selection metric?
**Answer:**  
Accuracy can hide class-specific errors. For safety-sensitive tasks, we must also inspect recall and false-negative rate.

---

## 32) What does confusion matrix mean?
**Answer:**  
It shows `TN, FP, FN, TP`.
- TP: fake correctly detected  
- FN: fake missed (dangerous)  
- FP: real flagged as fake  
- TN: real correctly accepted

---

## 33) What is balanced accuracy?
**Answer:**  
Average of class-wise recall. Better than plain accuracy for imbalanced data.

---

## 34) Which is more important for deepfake safety: precision or recall?
**Answer:**  
Depends on business need.  
- High recall: catch more deepfakes (fewer misses)  
- High precision: fewer false accusations on real media

---

## 35) What is latency target in your PRD?
**Answer:**  
Image prediction under ~3s and video under ~10s (with current frame sampling setup).

---

## 36) Explain Keras `.keras` vs `.h5` vs TensorFlow SavedModel vs `.pkl`
**Answer:**  
- `.keras`: modern Keras format (recommended for Keras models)  
- `.h5`: older Keras HDF5 format, still common  
- `SavedModel`: TensorFlow-native export format, great for serving and interoperability  
- `.pkl` / pickle: Python object serialization; useful for scikit-learn models, **not ideal for Keras neural networks**

**Rule of thumb:**  
Use `.keras` or `SavedModel` for deep learning, use `.pkl` mostly for sklearn pipelines.

---

## 37) Why is `.pkl` risky for deep models?
**Answer:**  
Pickle can be less portable across environments, more fragile with version changes, and unsafe with untrusted files.

---

## 38) What is ONNX and when use it?
**Answer:**  
ONNX is an interoperable model format for cross-framework deployment and optimization. Useful when serving outside Python/TensorFlow stack.

---

## 39) Why does version compatibility matter for model loading?
**Answer:**  
Saved model config can include keys unsupported by newer/older runtime versions.

**Real-time example:**  
Legacy keys like `renorm_*` or `quantization_config` caused deserialization errors in backend.

---

## 40) What immediate checklist do you follow before saying model is production-ready?
**Answer:**  
1. Leakage-free split verified  
2. External unseen test performance acceptable  
3. Metrics beyond accuracy validated  
4. Inference preprocessing exactly matches training pipeline  
5. Backend model loading tested on deployment environment  
6. Latency measured under expected load

---

## 41) How do you explain model confidence to non-technical users?
**Answer:**  
Confidence is how strongly model believes the predicted class, not a guarantee of truth. It should be combined with human review for critical cases.

---

## 42) What is model drift and how to handle it?
**Answer:**  
Drift means real-world data changes over time.  
Solution: monitor live metrics, collect new data, retrain periodically, and recalibrate threshold.

---

## 43) Why did you add dashboard visualizations?
**Answer:**  
To help non-ML users understand model behavior through confusion matrix, ROC curve, trend bars, and summary tables.

---

## 44) If viva asks “How will you reach 90% reliably?”, what is the answer?
**Answer:**  
Use leakage-safe split, increase data diversity, add external holdout, tune threshold on validation only, compare multiple backbones, and select by robust metrics (not single run accuracy).

---

## 45) One-line honest viva summary
**Answer:**  
“Our project is technically complete end-to-end, and we improved training/evaluation rigor; final trustworthy quality depends on leakage-free splitting and external unseen validation.”


---

## 46) Which algorithm is used in your current DeepShield pipeline?
**Answer:**  
We use deep learning classification with transfer learning. Backbones include `MobileNetV2`, `EfficientNetB0`, `ResNet50`, and `Xception`.

---

## 47) Which model gave best accuracy in your comparison runs?
**Answer:**  
In current dashboard comparisons, `ResNet50` appears as top performer, followed by `Xception`, `EfficientNetB0`, and `MobileNetV2` depending on the run and split strategy.

---

## 48) Why compare multiple backbones instead of one model?
**Answer:**  
Because every backbone has a tradeoff between speed, memory, and accuracy. Comparing models helps pick the best one for your hardware and target metric.

---

## 49) Why is MobileNetV2 still important if it is less accurate?
**Answer:**  
MobileNetV2 is lightweight and fast, useful for low-resource systems and real-time inference constraints.

---

## 50) Why can ResNet50 or Xception outperform MobileNetV2?
**Answer:**  
They are deeper/heavier architectures with stronger feature capacity, so they can learn complex deepfake artifacts better when data quality is good.

---

## 51) What does training from `_raw` mean in your project?
**Answer:**  
`_raw` contains original collected data. Scripts first prepare a balanced split from this source, then train on `train`, tune on `val`, and evaluate on `test`.

---

## 52) Why is split strategy (`group` vs `frame`) important?
**Answer:**  
`group` split reduces leakage by keeping related samples in one split. `frame` split may inflate metrics because nearby frames from same source can appear in both train and test.

---

## 53) What is leakage in one sentence?
**Answer:**  
Leakage is when test information indirectly reaches training, making reported performance unrealistically high.

---

## 54) How does `train_model_suite.py` help your workflow?
**Answer:**  
It automates full pipeline: prepare subset, train multiple backbones, evaluate each, rank results, and publish winner + comparison JSON for dashboard.

---

## 55) Which training hyperparameters are most important to mention in viva?
**Answer:**  
`epochs`, `fine-tune-epochs`, `batch-size`, `learning-rate`, `backbone`, `dropout`, `label-smoothing`, `monitor-metric`, and split strategy.

---

## 56) Why do you use EarlyStopping?
**Answer:**  
To stop training when validation metric stops improving, reducing overfitting and unnecessary compute.

---

## 57) Why do you use ReduceLROnPlateau?
**Answer:**  
It lowers learning rate when progress stalls, helping model converge to a better local optimum.

---

## 58) What is the role of `class_weights`?
**Answer:**  
Class weights compensate imbalance so minority class errors are penalized more during training.

---

## 59) Why not use only accuracy for model selection?
**Answer:**  
Accuracy can hide class-specific failures. We also check `F1`, `Recall`, `AUC`, and confusion matrix.

---

## 60) How do you explain Precision and Recall in this project context?
**Answer:**  
- Precision: when model says deepfake, how often it is correct.  
- Recall: how many actual deepfakes the model successfully catches.

---

## 61) Why tune threshold after training?
**Answer:**  
Threshold controls decision boundary. A tuned threshold can better align with business goal (for example, fewer false negatives).

---

## 62) Why should threshold be tuned on validation, not test?
**Answer:**  
Using test for tuning leaks evaluation data and inflates final reported performance.

---

## 63) What model artifacts are produced after training?
**Answer:**  
Common outputs: `best_model.keras`, `final_model.keras`, `metrics.json`, `history.json`, and comparison files.

---

## 64) Difference between `best_model.keras` and `final_model.keras`?
**Answer:**  
`best_model.keras` is best checkpoint by monitored validation metric; `final_model.keras` is model state at training end.

---

## 65) Difference between `.keras`, `SavedModel`, `.h5`, and `.pkl` in simple terms?
**Answer:**  
- `.keras`: preferred Keras model format  
- `SavedModel`: TensorFlow serving format  
- `.h5`: older Keras format  
- `.pkl`: pickle object format (mainly sklearn), not ideal primary format for deep Keras networks.

---

## 66) Why do version mismatches break model loading?
**Answer:**  
Serialized configs can contain keys unsupported in another Keras/TensorFlow version (for example legacy BN or quantization fields).

---

## 67) What is a practical sign of overfitting in your logs?
**Answer:**  
Training metrics improve while validation metrics stop improving or degrade.

---

## 68) If viva asks: "How will you make results trustworthy?"
**Answer:**  
Use leakage-safe split, external unseen test set, multi-metric reporting, fixed reproducible setup, and deployment-time validation.

---

## 69) If viva asks: "How will you improve model further?"
**Answer:**  
Increase data diversity, clean labels, add hard negatives, try stronger backbones, perform proper threshold calibration, and retrain periodically.

---

## 70) One-line technical summary for examiners
**Answer:**  
DeepShield uses transfer-learning CNN backbones with leakage-aware evaluation, threshold tuning, and dashboard-based multi-metric comparison for practical deepfake detection.

---

## 71) What train/val/test percentages did you use, and why?
**Answer:**  
We used approximately **70% train, 15% validation, 15% test**.

- **Train (70%)**: to learn model weights from enough examples  
- **Validation (15%)**: to tune hyperparameters and threshold (without touching test)  
- **Test (15%)**: final unbiased performance check

**Why this split?**  
It gives a good balance between learning capacity and reliable evaluation for medium-sized datasets.

**Important note in our project:**  
We prefer **group-based split** (not random frame split) so related frames from the same source do not leak across train/val/test. This makes reported accuracy more realistic.

---

## 72) Which models did you use in this project?
**Answer:**  
We used these deep learning backbones:
- `MobileNetV2`
- `EfficientNetB0`
- `ResNet50`
- `Xception`

These were compared to balance speed, model size, and accuracy.

---

## 73) Which algorithm did you use for training?
**Answer:**  
We used **CNN Transfer Learning for binary classification** (`authentic` vs `deepfake`).

Training setup:
- Optimizer: `Adam`
- Loss: `BinaryCrossentropy`
- Learning strategy: warmup + fine-tuning
- Evaluation: `Accuracy`, `Precision`, `Recall`, `F1`, `ROC-AUC`, `Specificity`, `Confusion Matrix`
