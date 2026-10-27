### Change 1 — cifar10 — 2026-10-24

**Change:** Log for the baseline

**Why:** N/A

**Expected:** Starting me logs with baseline

**Result:** Validation accuracy 46.86%; train accuracy 80.92%; validation loss 2.0368; train loss 0.5249; 1.24 min on cuda. Train/validation accuracy gap: 34.06%.

**Loss curve:** Epoch loss 1.8200 → 0.4724 over 30 logged epochs.

**Interpretation / next step:** 

**Reproducibility:** Package `residual_mlp_baseline`, seed `0`, train count `45000`, validation count `5000`, run ID `20261024_152702_baseline_residual_mlp_baseline`. 

### Change 2 — cifar10 — 2026-10-24

**Change:** Switched from tanh to GeLU for all residual blocks

**Why:** A different activation function seemed like a good starting point.

**Expected:** I expect near identical results compared to tanh. If not, then will explore why.

**Result:** Validation accuracy 49.72%; train accuracy 93.82%; validation loss 2.7147; train loss 0.1816; 1.23 min on cuda. Train/validation accuracy gap: 44.10%.

**Loss curve:** Epoch loss 1.7239 → 0.1040 over 30 logged epochs. Not converged yet

**Interpretation / next step:** This was unexpected. The model achieved significantly higher accuracy and lower loss on the train set but performed worse on the validaiton set. This suggests that model overfit and learned noise. It does have higher train and validation, which suggests it is still better than tanh. I would like to explore regularization techniques next. 

**Reproducibility:** Package `Residual_mlp_GeLU`, seed `0`, train count `45000`, validation count `5000`, run ID `20261024_153540_baseline_Residual_mlp_GeLU`. GeLU

### Change 3 — cifar10 — 2026-10-24

**Change:** Was happy with GeLU. Added weight decay of 1e-4

**Why:** Model was overfitting. Regularization made sense

**Expected:** I expect explanation gap to shrink. Better metric scores. 

**Result:** Validation accuracy 50.20%; train accuracy 95.88%; validation loss 2.5651; train loss 0.1319; 1.30 min on cuda. Train/validation accuracy gap: 45.68%.

**Loss curve:** Epoch loss 1.7238 → 0.1487 over 30 logged epochs. Looked to be converging

**Interpretation / next step:** Interestingly, every single metric improved except for the accuracy gap. The model understands the training data extremely well, but it is not generalizing. I'm sure I could get the training accuracy even higher, and maybe bump up the validation a few points, but I primarily need to address generalizaiton issues. Regularization isn't the problem here. Next step is some kind of data augmentation. I will keep weight decay for now. 

**Reproducibility:** Package `GeLU_WeightDecay`, seed `0`, train count `45000`, validation count `5000`, run ID `20261024_155353_baseline_GeLU_WeightDecay` GeLU, WD

### Change 4 — cifar10 — 2026-10-24

**Change:** Shifting to data Augmentation. First is horizontal flipping of 5% (still keeping original)

**Why:** Model was great on train set, but failed on validation. Needed to force model to generalize

**Expected:** I really have no idea. Hopefully better generalization gap. 

**Result:** Validation accuracy 50.14%; train accuracy 96.32%; validation loss 2.6165; train loss 0.1200; 1.32 min on cuda. Train/validation accuracy gap: 46.18%.

**Loss curve:** Epoch loss 1.7113 → 0.1932 over 30 logged epochs. Did not appear to converge just yet

**Interpretation / next step:** It is slightly worse than before though it's hard to say why. This was very small scale, so I will augment 15% this time. 

**Reproducibility:** Package `GeLU_WD_DataAug1`, seed `0`, train count `45000`, validation count `5000`, run ID `20261024_162119_baseline_GeLU_WD_DataAug1`. GeLU, WD, 5% horizontal flipping

### Change 5 — cifar10 — 2026-10-24

**Change:** Increased horizontal flipping form 5% to 15%

**Why:** Wanted to see effects of a larger augmentation.

**Expected:** I expect slightly worse performance because 5% decreased metrics. 

**Result:** Validation accuracy 50.36%; train accuracy 91.45%; validation loss 2.7493; train loss 0.2690; 1.46 min on cuda. Train/validation accuracy gap: 41.09%. 

**Loss curve:** Epoch loss 1.7029 → 0.1623 over 30 logged epochs. Close to converging

**Interpretation / next step:** Notably the train accuracy has dropped. I firmly beleive that I can have a 30 block network that can learn CIFAR10. I will try out another augmentation form before increasing capacity. 

**Reproducibility:** Package `GeLU_WD_DataAug2`, seed `0`, train count `45000`, validation count `5000`, run ID `20261024_162835_baseline_GeLU_WD_DataAug2`. GeLU, WD, 15% horizontal flipping

### Change 6 — cifar10 — 2026-10-24

**Change:** Shifting to Gaussian instead of horizontal flipping

**Why:** Flipping failed to produce results. I still believe in data augmentation

**Expected:** I expect something similar to 5% horizontal flipping

**Result:** Validation accuracy 50.20%; train accuracy 95.88%; validation loss 2.5651; train loss 0.1319; 1.26 min on cuda. Train/validation accuracy gap: 45.68%.

**Loss curve:** Epoch loss 1.7238 → 0.1487 over 30 logged epochs.

**Interpretation / next step:** The results are the same as 5% horizontal flipping. I do not expect data augmentation to solve the generalization issue until the model has greater capacity or inferential bias. 

**Reproducibility:** Package `GeLU_WD_DataAug3`, seed `0`, train count `45000`, validation count `5000`, run ID `20261024_164215_baseline_GeLU_WD_DataAug3`. GeLU, WD, 5% gaussian noise

### Change 7 — cifar10 — 2026-10-24

**Change:** Changed to patching architecture

**Why:** Additional data augmentation did not help. Already achieved high train metrics and needed to try something new.

**Expected:** I'm expecting better generalization, so a lower accuracy gap.

**Result:** Validation accuracy 52.56%; train accuracy 88.01%; validation loss 2.1935; train loss 0.3360; 2.51 min on cuda. Train/validation accuracy gap: 35.45%.

**Loss curve:** Epoch loss 2.0380 → 0.1802 over 30 logged epochs. Did not converge

**Interpretation / next step:** Well the accuracy gap dropped but so did training accuracy. At this point it seems like the model needs either more capacity or epochs. Before adding either, I will try a different optimizer. AdamW

**Reproducibility:** Package `GeLU_WD_Patch1`, seed `0`, train count `45000`, validation count `5000`, run ID `20261024_213150_baseline_GeLU_WD_Patch1`. GeLU, WD, patching 

### Change 8 — cifar10 — 2026-10-24

**Change:** Added AdamW to patch architecture

**Why:** The loss curve did not converge, hopefully a different optimizer will allow us to get more out of the model.

**Expected:** I'm expecting better metrics and a converging loss curve.

**Result:** Validation accuracy 59.54%; train accuracy 98.38%; validation loss 2.6181; train loss 0.0478; 2.46 min on cuda. Train/validation accuracy gap: 38.84%.

**Loss curve:** Epoch loss 1.6427 → 0.0545 over 30 logged epochs. Curve converged.

**Interpretation / next step:** AdamW improved learning, which allowed us to squeeze more out of the model. Because we have hit 98%+ accuracy, and there is still a gap of 38% I think now is a good time to return to data augmentation. The optiminzation portion is done for now.

**Reproducibility:** Package `GeLU_WD_Patch2`, seed `0`, train count `45000`, validation count `5000`, run ID `20261024_214800_baseline_GeLU_WD_Patch2`. GeLU, WD, patching, AdamW

### Change 9 — cifar10 — 2026-10-24

**Change:** Added 10% horizontal flipping and gaussian noise

**Why:** The model is optimized for the current size and architecture. Further generalization will come from better data.

**Expected:** I expect higher validation accuracy.

**Result:** Validation accuracy 61.54%; train accuracy 98.80%; validation loss 2.5840; train loss 0.0348; 2.80 min on cuda. Train/validation accuracy gap: 37.26%.

**Loss curve:** Epoch loss 1.6006 → 0.0438 over 30 logged epochs. Converged

**Interpretation / next step:** The higher validaiton suggests that the model generalized better with augmented data. I shall add more. 

**Reproducibility:** Package `GeLU_WD_DA_Patch1`, seed `0`, train count `45000`, validation count `5000`, run ID `20261024_215942_baseline_GeLU_WD_DA_Patch1`. GeLU, WD, patching, AdamW, 10% H-flip, 10% G-noise
### Change 10 — cifar10 — 2026-10-24

**Change:** Removed 10% h-flipping

**Why:** Check how much of improvement is from h-flipping or g-noise.

**Expected:** I expected lowered validation accuracy.

**Result:** Validation accuracy 59.34%; train accuracy 97.82%; validation loss 2.5241; train loss 0.0634; 2.66 min on cuda. Train/validation accuracy gap: 38.48%.

**Loss curve:** Epoch loss 1.6350 → 0.0826 over 30 logged epochs. Converged

**Interpretation / next step:** Guassian noise doesn't help much. H-flipping was useful. Next I will try sorbel augmentation,  

**Reproducibility:** Package `GeLU_WD_DA_Patch2`, seed `0`, train count `45000`, validation count `5000`, run ID `20261024_221344_baseline_GeLU_WD_DA_Patch2`. GeLU, WD, patching, AdamW, 10% G-noise

### Change 11 — cifar10 — 2026-10-24

**Change:** Sobel filtering 10%

**Why:** I already tested two augmentation methods. Wanted to try a third.

**Expected:** I have no expectation

**Result:** Validation accuracy 61.60%; train accuracy 98.37%; validation loss 2.4154; train loss 0.0481; 2.64 min on cuda. Train/validation accuracy gap: 36.77%.

**Loss curve:** Epoch loss 1.7081 → 0.0749 over 30 logged epochs. Converged

**Interpretation / next step:** Sobel seemed to be just as useful as h-flipping. Will try higher percentage of both. 

**Reproducibility:** Package `GeLU_WD_DA_Patch3`, seed `0`, train count `45000`, validation count `5000`, run ID `20261024_222743_baseline_GeLU_WD_DA_Patch3`. GeLU, WD, patching, AdamW, 10% Sobel

### Change 12 — cifar10 — 2026-10-24

**Change:** 15% Sobel, H-flip, g-noise

**Why:** I found three methods, now next step is increasing amount

**Expected:** I'm expecting better generalization.

**Result:** Validation accuracy 61.22%; train accuracy 97.76%; validation loss 2.5585; train loss 0.0656; 3.45 min on cuda. Train/validation accuracy gap: 36.54%.

**Loss curve:** Epoch loss 1.6371 → 0.0619 over 30 logged epochs.

**Interpretation / next step:** Further Augmentation did not improve results. Will remove augmentation for now and adjust patch size. 

**Reproducibility:** Package `GeLU_WD_DA_Patch4`, seed `0`, train count `45000`, validation count `5000`, run ID `20261024_223740_baseline_GeLU_WD_DA_Patch4`. GeLU, WD, patching, AdamW, 15% Sobel H-Flipping G-noise


### Change 13 — cifar10 — 2026-10-24

**Change:** 2x2 patch size from 4x4 

**Why:** It's another dimension to explore.

**Expected:** I expect longer training times and higher accuracy.

**Result:** Validation accuracy 59.46%; train accuracy 98.08%; validation loss 2.6055; train loss 0.0557; 7.72 min on cuda. Train/validation accuracy gap: 38.62%.

**Loss curve:** Epoch loss 1.6461 → 0.0781 over 30 logged epochs. Converged

**Interpretation / next step:** Well accuracy was basically unchanged and training time more than doubled. So kind of a wash. I will try 8x8 next. Short training time would be nice if accuracy does not drop. 

**Reproducibility:** Package `GeLU_WD_Patch_Adj1`, seed `0`, train count `45000`, validation count `5000`, run ID `20261024_225558_baseline_GeLU_WD_Patch_Adj1`.  GeLU, WD, patching (2x2), AdamW

### Change 14 — cifar10 — 2026-10-24

**Change:** 8x8 patch size

**Why:** Want to decrease computation and maintain accuraccy

**Expected:** reduced accuracy

**Result:** Validation accuracy 59.04%; train accuracy 98.14%; validation loss 2.8159; train loss 0.0531; 1.71 min on cuda. Train/validation accuracy gap: 39.10%.

**Loss curve:** Epoch loss 1.6683 → 0.0476 over 30 logged epochs.

**Interpretation / next step:** Slight accuracy loss. Will maintain patch size while exploring adding more capacity.

**Reproducibility:** Package `GeLU_WD_Patch_Adj2`, seed `0`, train count `45000`, validation count `5000`, run ID `20261024_230241_baseline_GeLU_WD_Patch_Adj2`. GeLU, WD, patching (8x8), AdamW

### Change 15 — cifar10 — 2026-10-24

**Change:** 8x8 patch size with 512 hidden

**Why:** Want to explore increased capacity now

**Expected:** Improved accuracy

**Result:** Validation accuracy 62.20%; train accuracy 98.80%; validation loss 2.5475; train loss 0.0365; 1.71 min on cuda. Train/validation accuracy gap: 36.60%.

**Loss curve:** Epoch loss 1.7143 → 0.0470 over 30 logged epochs. Converged

**Interpretation / next step:** Accuracy Improved. Additional hidden units helped. Will now explore a wider residual space.

**Reproducibility:** Package `GeLU_WD_8x8_Capacity1`, seed `0`, train count `45000`, validation count `5000`, run ID `20261024_231727_baseline_GeLU_WD_8x8_Capacity1`. GeLU, WD, patching (8x8), AdamW, 512 hidden

### Change 16 — cifar10 — 2026-10-24

**Change:** 8x8 patch size with 512 hidden & 256 wide

**Why:** Further capacity change exploration

**Expected:** Improved accuracy

**Result:** Validation accuracy 62.04%; train accuracy 99.22%; validation loss 2.5067; train loss 0.0231; 2.05 min on cuda. Train/validation accuracy gap: 37.18%.

**Loss curve:** Epoch loss 1.8027 → 0.0353 over 30 logged epochs.

**Interpretation / next step:** Extra width did not improve results. I will revert back to 128 and instead increase the patch hidden capacity.

**Reproducibility:** Package `GeLU_WD_8x8_Capacity2`, seed `0`, train count `45000`, validation count `5000`, run ID `20261024_232258_baseline_GeLU_WD_8x8_Capacity2`. GeLU, WD, patching (8x8), AdamW, 512 hidden, 256 width

### Change 17 — cifar10 — 2026-10-25

**Change:** 8x8 patch size with 512 hidden & 256 patch hidden

**Why:** Want to explore increased capacity now

**Expected:** Improved accuracy

**Result:** Validation accuracy 62.32%; train accuracy 98.92%; validation loss 2.6155; train loss 0.0303; 1.95 min on cuda. Train/validation accuracy gap: 36.60%.

**Loss curve:** Epoch loss 1.7225 → 0.0380 over 30 logged epochs.

**Interpretation / next step:** Very little change from the added capacity. I'm going to take a step back from additional capacity and explore adding a scheduler and dropout. I want to avoid adding more layers for now.

**Reproducibility:** Package `GeLU_WD_8x8_Capacity3`, seed `0`, train count `45000`, validation count `5000`, run ID `20261025_103240_baseline_GeLU_WD_8x8_Capacity3`. GeLU, WD, patching (8x8), AdamW, 512 hidden, 256 patch hidden

### Change 18 — cifar10 — 2026-10-25

**Change:** Added cosine annealing scheduler

**Why:** Additional capacity apart from more layers did not help. Want to explore further optimization before adding more layers

**Expected:** Improved accuracy

**Result:** Validation accuracy 63.50%; train accuracy 100.00%; validation loss 2.9550; train loss 0.0001; 1.78 min on cuda. Train/validation accuracy gap: 36.50%.

**Loss curve:** Epoch loss 1.7143 → 0.0001 over 30 logged epochs. converged

**Interpretation / next step:** Wow. So There is now perfect train accuracy and higher validation accuracy. One important point is that the higher validaiton loss. So the model is now much more confidently wrong in some cases. Dropout is typically used to address this issue so i will try this next. 

**Reproducibility:** Package `GeLU_WD_8x8_Scheduler`, seed `0`, train count `45000`, validation count `5000`, run ID `20261025_105103_baseline_GeLU_WD_8x8_Scheduler`. GeLU, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 

### Change 19 — cifar10 — 2026-10-25

**Change:** Added p=0.1 Dropout in standard/non patch mixing blocks

**Why:** The model has memorized the train set and is not generalizing well. Dropout might help

**Expected:** Well hopefully the model still learns the train set perfectly and generalizes to the test set

**Result:** Validation accuracy 65.06%; train accuracy 100.00%; validation loss 2.7115; train loss 0.0001; 1.88 min on cuda. Train/validation accuracy gap: 34.94%.

**Loss curve:** Epoch loss 1.7196 → 0.0015 over 30 logged epochs. converged

**Interpretation / next step:** 10% dropout improved generalization. I will increase to 20%

**Reproducibility:** Package `GeLU_WD_CAS_8x8_DropO`, seed `0`, train count `45000`, validation count `5000`, run ID `20261025_110700_baseline_GeLU_WD_CAS_8x8_DropO`. GeLU, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 10% dropout

### Change 20 — cifar10 — 2026-10-25

**Change:** Added p=0.2 Dropout in standard/non patch mixing blocks

**Why:** Dropout helped. Adding more should continue to help. 

**Expected:** I expect validaiton accuracy to improve.

**Result:** Validation accuracy 67.22%; train accuracy 100.00%; validation loss 2.3624; train loss 0.0002; 1.88 min on cuda. Train/validation accuracy gap: 32.78%.

**Loss curve:** Epoch loss 1.7280 → 0.0066 over 30 logged epochs. Converged

**Interpretation / next step:** More dropout helped. I want to add data augmentation before I explore more dropout. I will see the effects of reintroducing data augmentation before introducing futher dropout. 

**Reproducibility:** Package `GeLU_WD_CAS_8x8_DropO2`, seed `0`, train count `45000`, validation count `5000`, run ID `20261025_111550_baseline_GeLU_WD_CAS_8x8_DropO2`. GeLU, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout
### Change 21 — cifar10 — 2026-10-25

**Change:** Added 50% horizontal flipping

**Why:** I wanted to add more augmented data before futher dropout. Better to make use of too much capacity than hinder it.

**Expected:** I expect higher validaiton accuracy

**Result:** Validation accuracy 69.50%; train accuracy 100.00%; validation loss 2.3585; train loss 0.0001; 3.59 min on cuda. Train/validation accuracy gap: 30.50%.

**Loss curve:** Epoch loss 1.6614 → 0.0023 over 30 logged epochs. Converged

**Interpretation / next step:** Horizontal flipping helped with generalization. I am a bit concerned about transfering this to imagenet, so instead of augmenting at the start, I will swap to on the fly augmentation and see the effects..

**Reproducibility:** Package `GeLU_WD_CAS_DA_8x8_DropO`, seed `0`, train count `45000`, validation count `5000`, run ID `20261025_112814_baseline_GeLU_WD_CAS_DA_8x8_DropO`. GeLU, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 50 H-Flipping

### Change 22 — cifar10 — 2026-10-25

**Change:** On the fly augmentation

**Why:** Concerned about memory usage when shifting to imagenet

**Expected:** Little to no change

**Result:** Validation accuracy 69.08%; train accuracy 100.00%; validation loss 2.3339; train loss 0.0001; 3.66 min on cuda. Train/validation accuracy gap: 30.92%.

**Loss curve:** Epoch loss 1.6614 → 0.0022 over 30 logged epochs. Converged

**Interpretation / next step:** This matched expectations. Now I will add 50% vertical flipping.

**Reproducibility:** Package `GeLU_WD_CAS_DA_8x8_DropO2`, seed `0`, train count `45000`, validation count `5000`, run ID `20261025_121023_baseline_GeLU_WD_CAS_DA_8x8_DropO2`. GeLU, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 50% H-Flipping

### Change 23 — cifar10 — 2026-10-25

**Change:** Added 50% vertical flipping

**Why:** 50% horizontal helped and vertical flipping seems like a real world possability

**Expected:** Improved validation accuracy

**Result:** Validation accuracy 69.74%; train accuracy 100.00%; validation loss 2.0946; train loss 0.0001; 4.79 min on cuda. Train/validation accuracy gap: 30.26%.

**Loss curve:** Epoch loss 1.6614 → 0.0063 over 30 logged epochs.

**Interpretation / next step:** Vertical flipping do not improve generalization. Additional compute may not be worth. 

**Reproducibility:** Package `GeLU_WD_CAS_DA_8x8_DropO3`, seed `0`, train count `45000`, validation count `5000`, run ID `20261025_123258_baseline_GeLU_WD_CAS_DA_8x8_DropO3`. GeLU, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 50% H/V-Flipping


### Change 24 — cifar10 — 2026-10-26

**Change:** Replaced vertical flipping with random erasure

**Why:** Trying out different data augmentation methods

**Expected:** Similar training results

**Result:** Validation accuracy 69.60%; train accuracy 100.00%; validation loss 1.9186; train loss 0.0001; 4.88 min on cuda. Train/validation accuracy gap: 30.40%.

**Loss curve:** Epoch loss 1.5843 → 0.0124 over 30 logged epochs. Converged

**Interpretation / next step:** This is basically the same as with vertical flipping, though notably loss decreased faster at the start. Next step is brightness adjustment. 

**Reproducibility:** Package `GeLU_WD_CAS_DA_8x8_DropO4`, seed `0`, train count `45000`, validation count `5000`, run ID `20261026_112448_baseline_GeLU_WD_CAS_DA_8x8_DropO4`. GeLU, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 50% H-Flipping/erasure

### Change 25 — cifar10 — 2026-10-26

**Change:** Incrased horizontal flipping to 100%

**Why:** Horizontal flipping helped tremendously before, why not max and see the effects.

**Expected:** Improved validationa ccuracy

**Result:** Validation accuracy 69.80%; train accuracy 100.00%; validation loss 1.9790; train loss 0.0001; 6.11 min on cuda. Train/validation accuracy gap: 30.20%.

**Loss curve:** Epoch loss 1.5151 → 0.0078 over 30 logged epochs. Converged

**Interpretation / next step:** This is what i expected, but the gain was negligible. Not worth extra compute. Will reduce back to 50%.

**Reproducibility:** Package `GeLU_WD_CAS_DA_8x8_DropO5`, seed `0`, train count `45000`, validation count `5000`, run ID `20261026_113649_baseline_GeLU_WD_CAS_DA_8x8_DropO5`.GeLU, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 100% H-Flipping, 50% erasure

### Change 26 — cifar10 — 2026-10-26

**Change:** Reduced down to 50% horizontal and added 50% +- 10 rotations

**Why:** Additional compute not worth. Switch to different augmentation method

**Expected:** Improved accuracy

**Result:** Validation accuracy 70.74%; train accuracy 100.00%; validation loss 1.8460; train loss 0.0001; 9.05 min on cuda. Train/validation accuracy gap: 29.26%.

**Loss curve:** Epoch loss 1.5093 → 0.0133 over 30 logged epochs. Converged

**Interpretation / next step:** An additional data augmentation helped. The increased training time is concerning though. I will reduce the percentage of augmentation across the board and see effects on training time and accuracy. 

**Reproducibility:** Package `GeLU_WD_CAS_DA_8x8_DropO6`, seed `0`, train count `45000`, validation count `5000`, run ID `20261026_115735_baseline_GeLU_WD_CAS_DA_8x8_DropO6`. GeLU, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 50% H-Flipping, 50% erasure, 50% +- 1- rotations.

### Change 27 — cifar10 — 2026-10-26

**Change:** reduced all forms of data augmentation to 25%

**Why:** Need to reduce training time.

**Expected:** Slightly reduced accuracy.

**Result:** Validation accuracy 70.36%; train accuracy 100.00%; validation loss 1.6301; train loss 0.0003; 4.88 min on cuda. Train/validation accuracy gap: 29.64%.

**Loss curve:** Epoch loss 1.6059 → 0.0276 over 30 logged epochs. Converged

**Interpretation / next step:** Additional forms of data augmentation allow for less overall augmentation. A drastic reduction barely reduced accuracy. I am done

**Reproducibility:** Package `GeLU_WD_CAS_DA_8x8_DropO7`, seed `0`, train count `45000`, validation count `5000`, run ID `20261026_120408_baseline_GeLU_WD_CAS_DA_8x8_DropO7`. GeLU, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 25% H-Flipping, 25% erasure, 25% +- 1- rotations.

