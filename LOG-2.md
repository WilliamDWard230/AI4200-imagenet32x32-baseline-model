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

### Change 28 — cifar10 — 2026-10-28

**Change:** Reduced data augmentation rates to 20%

**Why:** Reducing down to 25% saved on compute and had little effect on accuracy, will drop as low as a I can.

**Expected:** Similar accuracy and reduced comptue time

**Result:** Validation accuracy 69.02%; train accuracy 100.00%; validation loss 1.6020; train loss 0.0006; 4.21 min on cuda. Train/validation accuracy gap: 30.98%.

**Loss curve:** Epoch loss 1.6547 → 0.0404 over 30 logged epochs.

**Interpretation / next step:** This was a bit more of a drop than i anticipated, but I still think the saved compute will be worth it. I think having a smaller percentage of a wide variety of augemntation methods will help more than a large percentage of a few, so i will keep it at 20% and add another method.

**Reproducibility:** Package `GeLU_WD_CAS_DA_8x8_DropO8`, seed `0`, train count `45000`, validation count `5000`, run ID `20261028_154738_baseline_GeLU_WD_CAS_DA_8x8_DropO8`. GeLU, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations.
### Change 29 — cifar10 — 2026-10-28

**Change:** Added 20% +- 20% brightness

**Why:** Hoping that another form of augmentation will improve generalization.

**Expected:** Similar accuracy and reduced comptue time

**Result:** Validation accuracy 70.02%; train accuracy 100.00%; validation loss 1.6296; train loss 0.0003; 4.68 min on cuda. Train/validation accuracy gap: 29.98%.

**Loss curve:** Epoch loss 1.6132 → 0.0293 over 30 logged epochs.

**Interpretation / next step:** This did not significantly increase compute time and regained most of the lost accuracy. I will add another form of data augmentation and see the additional effect. 

**Reproducibility:** Package `GeLU_WD_CAS_DA_8x8_DropO9`, seed `0`, train count `45000`, validation count `5000`, run ID `20261028_160242_baseline_GeLU_WD_CAS_DA_8x8_DropO9`.  GeLU, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% +-Brightness.

### Change 30 — cifar10 — 2026-10-28

**Change:** Added 20% +- 2 shift pixel

**Why:** Attempting to improve generalization

**Expected:** Improved accuracy for some additional compute time

**Result:** Validation accuracy 72.08%; train accuracy 100.00%; validation loss 1.3611; train loss 0.0008; 5.33 min on cuda. Train/validation accuracy gap: 27.92%.

**Loss curve:** Epoch loss 1.5788 → 0.0646 over 30 logged epochs.

**Interpretation / next step:** This did drastically improve generalization (2%), but the added compute time is concerning. I will increase shift to +-3

**Reproducibility:** Package `GeLU_WD_CAS_DA_8x8_DropO10`, seed `0`, train count `45000`, validation count `5000`, run ID `20261028_162231_baseline_GeLU_WD_CAS_DA_8x8_DropO10`. GeLU, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% +-Brightness, 20% +- 2 pixel shift.

### Change 31 — cifar10 — 2026-10-28

**Change:** changed to +- 3 pixel shift

**Why:** More drastic shifts may improve generalization

**Expected:** Better generalization

**Result:** Validation accuracy 72.32%; train accuracy 100.00%; validation loss 1.3457; train loss 0.0010; 5.33 min on cuda. Train/validation accuracy gap: 27.68%.

**Loss curve:** Epoch loss 1.5808 → 0.0862 over 30 logged epochs.

**Interpretation / next step:** Very little change, though slight improvement. I will keep at +-3 for now. I will now try out a different activation function.

**Reproducibility:** Package `GeLU_WD_CAS_DA_8x8_DropO11`, seed `0`, train count `45000`, validation count `5000`, run ID `20261028_163606_baseline_GeLU_WD_CAS_DA_8x8_DropO11`. GeLU, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% +-Brightness, 20% +- 3 pixel shift.

### Change 32 — cifar10 — 2026-10-28

**Change:** Changed activation to SiLU

**Why:** Exploring my options.

**Expected:** I have no expectations.

**Result:** Validation accuracy 72.92%; train accuracy 100.00%; validation loss 1.3331; train loss 0.0022; 5.35 min on cuda. Train/validation accuracy gap: 27.08%.

**Loss curve:** Epoch loss 1.6086 → 0.1033 over 30 logged epochs.

**Interpretation / next step:** One intersting observation is that higher loss has typically translate to better generalization. SiLU improved generalization a bit. I think i will add a beta of 0.5 and switch to swish. 

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO`, seed `0`, train count `45000`, validation count `5000`, run ID `20261028_164608_baseline_SiLU_WD_CAS_DA_8x8_DropO`. SiLU, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% +-Brightness, 20% +- 3 pixel shift.

### Change 33 — cifar10 — 2026-10-28

**Change:** Changed activation to swish with beta of 0.5

**Why:** Exploring my options.

**Expected:** Slightly lower loss and perhaps better generalization or even worse

**Result:** Validation accuracy 72.08%; train accuracy 99.96%; validation loss 1.1676; train loss 0.0148; 5.53 min on cuda. Train/validation accuracy gap: 27.88%.

**Loss curve:** Epoch loss 1.6820 → 0.1722 over 30 logged epochs.

**Interpretation / next step:** Loss and accuracy were worse. An aggressive beta did not help. Will switch to 0.85 to see middle ground change from SiLU

**Reproducibility:** Package `Swish_p5_WD_CAS_DA_8x8_DropO`, seed `0`, train count `45000`, validation count `5000`, run ID `20261028_170012_baseline_Swish_p5_WD_CAS_DA_8x8_DropO`. Swish 0.5, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% +-Brightness, 20% +- 3 pixel shift.

### Change 34 — cifar10 — 2026-10-28

**Change:** Changed beta to 0.85

**Why:** Exploring my options.

**Expected:** Slightly lower loss and higher accuracy

**Result:** Validation accuracy 72.60%; train accuracy 100.00%; validation loss 1.2476; train loss 0.0031; 5.56 min on cuda. Train/validation accuracy gap: 27.40%.

**Loss curve:** Epoch loss 1.6214 → 0.1134 over 30 logged epochs.

**Interpretation / next step:** A beta value closer to silu seems to do better. will revert to silu. Will look into less depth.

**Reproducibility:** Package `Swish_p85_WD_CAS_DA_8x8_DropO`, seed `0`, train count `45000`, validation count `5000`, run ID `20261028_171001_baseline_Swish_p85_WD_CAS_DA_8x8_DropO`. Swish 0.85, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% +-Brightness, 20% +- 3 pixel shift.

### Change 35 — cifar10 — 2026-10-28

**Change:** Changed back to Silu and reduced depth to 12

**Why:** Exploring my options.

**Expected:** worse

**Result:** Validation accuracy 72.66%; train accuracy 100.00%; validation loss 1.2489; train loss 0.0032; 4.77 min on cuda. Train/validation accuracy gap: 27.34%.

**Loss curve:** Epoch loss 1.6157 → 0.1230 over 30 logged epochs.

**Interpretation / next step:** Reduced depth seems fine. Basically no change in accuracy, and significantly less compute. Got the goahead to use gated blocks, so i will be switching to SwiGLU. everything else will be the same

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO1`, seed `0`, train count `45000`, validation count `5000`, run ID `20261028_172140_baseline_SiLU_WD_CAS_DA_8x8_DropO1`. SiLU , 12 C-Blocks, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% +-Brightness, 20% +- 3 pixel shift.


### Change 36 — cifar10 — 2026-10-28

**Change:** Added 20% 0.8-1.2 contrast augmentation

**Why:** Model perfectly predicts training, but fails to generalize, i guess i need more augmentation to help

**Expected:** better accuracy and more compute

**Result:** Validation accuracy 73.04%; train accuracy 100.00%; validation loss 1.2047; train loss 0.0018; 7.72 min on cuda. Train/validation accuracy gap: 26.96%.

**Loss curve:** Epoch loss 1.5878 → 0.1039 over 30 logged epochs.

**Interpretation / next step:** Model still perfectly predicts training. Will add more augmentation. 

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO2`, seed `0`, train count `45000`, validation count `5000`, run ID `20261028_215007_baseline_SiLU_WD_CAS_DA_8x8_DropO2`. SiLU , 12 C-Blocks, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% +-Brightness, 20% +- 3 pixel shift, 20% 0.8-1.2 contrast augmentation.

### Change 37 — cifar10 — 2026-10-28

**Change:** Added 20% 0.8-1.2 color saturation augmentation

**Why:** Model needs to generalize. Augmentation is all i can think of to help.

**Expected:** better accuracy and more compute

**Result:** Validation accuracy 72.64%; train accuracy 100.00%; validation loss 1.3178; train loss 0.0015; 9.65 min on cuda. Train/validation accuracy gap: 27.36%.

**Loss curve:** Epoch loss 1.5775 → 0.1004 over 30 logged epochs.

**Interpretation / next step:** Pretty significant jump in validation loss. I guess this was a bad change. Will revert to SiLU_WD_CAS_DA_8x8_DropO2 and add a different augmentation

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO3`, seed `0`, train count `45000`, validation count `5000`, run ID `20261028_220611_baseline_SiLU_WD_CAS_DA_8x8_DropO3`. SiLU , 12 C-Blocks, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% +-Brightness, 20% +- 3 pixel shift, 20% 0.8-1.2 contrast augmentation, 20% 0.8-1.2 color saturation augmentation.

### Change 38 — cifar10 — 2026-10-28

**Change:** Added 20% cropping and resizing

**Why:** Model needs to generalize. Augmentation is all i can think of to help.

**Expected:** better accuracy and more compute

**Result:** Validation accuracy 73.36%; train accuracy 100.00%; validation loss 1.2972; train loss 0.0014; 9.32 min on cuda. Train/validation accuracy gap: 26.64%.

**Loss curve:** Epoch loss 1.5815 → 0.0973 over 30 logged epochs.

**Interpretation / next step:** The validaiton loss did not substantially improve though accuracy is slightly better. compute time may not be worth. 

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO4`, seed `0`, train count `45000`, validation count `5000`, run ID `20261028_222236_baseline_SiLU_WD_CAS_DA_8x8_DropO4`. SiLU , 12 C-Blocks, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% +-Brightness, 20% +- 3 pixel shift, 20% 0.8-1.2 contrast augmentation, 20% crop and resize
### Change 39 — cifar10 — 2026-10-28

**Change:** changed augmentations to 30% and increased batach size to 512

**Why:** I can't think of any more augmentations to add. Will increase batch size to add more augmentation %

**Expected:** No idea

**Result:** Validation accuracy 73.70%; train accuracy 100.00%; validation loss 1.3031; train loss 0.0012; 8.64 min on cuda. Train/validation accuracy gap: 26.30%.

**Loss curve:** Epoch loss 1.5639 → 0.0937 over 30 logged epochs.

**Interpretation / next step:** Slight improvement.

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO5`, seed `0`, train count `45000`, validation count `5000`, run ID `20261028_224223_baseline_SiLU_WD_CAS_DA_8x8_DropO5`. SiLU , 12 C-Blocks, 512 batch size, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 30% H-Flipping, 30% erasure, 30% +- 1- rotations, 30% +-Brightness, 30% +- 3 pixel shift, 30% 0.8-1.2 contrast augmentation, 30% crop and resize

### Change 40 — cifar10 — 2026-10-28

**Change:** Increased augmentation to 40% and batch size to 1024

**Why:** Worked last time, lets see if the trend continues

**Expected:** better accuracy and less compute

**Result:** Validation accuracy 73.80%; train accuracy 100.00%; validation loss 1.2077; train loss 0.0019; 10.20 min on cuda. Train/validation accuracy gap: 26.20%.

**Loss curve:** Epoch loss 1.5869 → 0.1150 over 30 logged epochs.

**Interpretation / next step:** This did not really provide futher benfit. Validation loss seems a bit high, so will try out a more aggressive starting LR

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO6`, seed `0`, train count `45000`, validation count `5000`, run ID `20261028_230038_baseline_SiLU_WD_CAS_DA_8x8_DropO6`. SiLU , 12 C-Blocks, 1024 batch size, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 40% H-Flipping, 40% erasure, 40% +- 1- rotations, 40% +-Brightness, 40% +- 3 pixel shift, 40% 0.8-1.2 contrast augmentation, 40% crop and resize

### Change 41 — cifar10 — 2026-10-28

**Change:** increased learning rate to 3e-3

**Why:** With fewer updates, will try more aggresive updates to compensate

**Expected:** better accuracy

**Result:** Validation accuracy 73.82%; train accuracy 100.00%; validation loss 1.3696; train loss 0.0003; 10.17 min on cuda. Train/validation accuracy gap: 26.18%.

**Loss curve:** Epoch loss 1.6869 → 0.0655 over 30 logged epochs.

**Interpretation / next step:**  This provided almost no benefit and actually incrased validaiton loss some. Will keep for now and explore increasing representation dimensions.

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO7`, seed `0`, train count `45000`, validation count `5000`, run ID `20261028_231227_baseline_SiLU_WD_CAS_DA_8x8_DropO7`. SiLU , 12 C-Blocks, 1024 batch size, LR: 3e-3, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 40% H-Flipping, 40% erasure, 40% +- 1- rotations, 40% +-Brightness, 40% +- 3 pixel shift, 40% 0.8-1.2 contrast augmentation, 40% crop and resize

### Change 42 — cifar10 — 2026-10-28

**Change:** Increased width and patch hidden (residual) from 128 to 256

**Why:** With fewer updates, will try more aggresive updates to compensate

**Expected:** better accuracy

**Result:** Validation accuracy 73.46%; train accuracy 100.00%; validation loss 1.5048; train loss 0.0001; 13.10 min on cuda. Train/validation accuracy gap: 26.54%.

**Loss curve:** Epoch loss 1.7541 → 0.0444 over 30 logged epochs.

**Interpretation / next step:** None of these past few changes have helped. I am going to revert back to SiLU_WD_CAS_DA_8x8_DropO1 and see if i can fine tune that any futher. 

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO8`, seed `0`, train count `45000`, validation count `5000`, run ID `20261028_232847_baseline_SiLU_WD_CAS_DA_8x8_DropO8`. 

### Change 43 — cifar10 — 2026-10-29

**Change:** reverted back to SiLU_WD_CAS_DA_8x8_DropO9 and increased batch size to 512

**Why:** I want to revert back to a working model and try to reduce compute without negatively impacting accuracy

**Expected:** less compute

**Result:** Validation accuracy 71.64%; train accuracy 100.00%; validation loss 1.1928; train loss 0.0056; 4.43 min on cuda. Train/validation accuracy gap: 28.36%.

**Loss curve:** Epoch loss 1.6211 → 0.1604 over 30 logged epochs.

**Interpretation / next step:** A pretty significant drop in accuracy, but three less minutes of training time. I can take that. I also notice that validation loss is lower. Perhaps now that the model is not as overly confident as it once was, I can adjust dropout. 

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO9`, seed `0`, train count `45000`, validation count `5000`, run ID `20261029_101239_baseline_SiLU_WD_CAS_DA_8x8_DropO9`. SiLU , 512 batch size, 12 C-Blocks, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% +-Brightness, 20% +- 3 pixel shift, 20% 0.8-1.2 contrast augmentation.

### Change 44 — cifar10 — 2026-10-29

**Change:** removed dropout

**Why:** I had dropout because model was confidently wrong. Now it is less confident but still wrong, perhaps additional parameters will help

**Expected:** worse generalization. I don't think it will work but i'm at my wits end

**Result:** Validation accuracy 70.44%; train accuracy 100.00%; validation loss 1.3380; train loss 0.0030; 4.32 min on cuda. Train/validation accuracy gap: 29.56%.

**Loss curve:** Epoch loss 1.6041 → 0.1039 over 30 logged epochs.

**Interpretation / next step:** Removing drop out failed. Will bring back and increase to 30%.

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO10`, seed `0`, train count `45000`, validation count `5000`, run ID `20261029_102126_baseline_SiLU_WD_CAS_DA_8x8_DropO10`. SiLU , 512 batch size, 12 C-Blocks, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% +-Brightness, 20% +- 3 pixel shift, 20% 0.8-1.2 contrast augmentation.

### Change 45 — cifar10 — 2026-10-29

**Change:** added and increased dropout to 30%

**Why:** Removing did not help, so perhaps increasing will.

**Expected:** Improved accuracy. 

**Result:** Validation accuracy 72.08%; train accuracy 99.98%; validation loss 1.1540; train loss 0.0115; 4.48 min on cuda. Train/validation accuracy gap: 27.90%.

**Loss curve:** Epoch loss 1.6355 → 0.2010 over 30 logged epochs.

**Interpretation / next step:** Interestingly train accuracy is no longer 100%. The validaiton accuracy score also climbed back up a bit. Will increase drop out to 40%. 

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO11`, seed `0`, train count `45000`, validation count `5000`, run ID `20261029_103000_baseline_SiLU_WD_CAS_DA_8x8_DropO11`. SiLU , 512 batch size, 12 C-Blocks, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 30% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% +-Brightness, 20% +- 3 pixel shift, 20% 0.8-1.2 contrast augmentation.

### Change 46 — cifar10 — 2026-10-29

**Change:** 40% dropout

**Why:** More dropout helped

**Expected:** Increased validation accuracy and lower training accuracy. Lower validaiton loss

**Result:** Validation accuracy 71.80%; train accuracy 99.93%; validation loss 1.1191; train loss 0.0223; 4.45 min on cuda. Train/validation accuracy gap: 28.13%.

**Loss curve:** Epoch loss 1.6532 → 0.2387 over 30 logged epochs.

**Interpretation / next step:** 40% dropout did not help. I'm going to return back to SiLU_WD_CAS_DA_8x8_DropO2. I think I can use a lower learning rate and more epochs to squeeze out better results. 

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO12`, seed `0`, train count `45000`, validation count `5000`, run ID `20261029_104205_baseline_SiLU_WD_CAS_DA_8x8_DropO12`. SiLU , 512 batch size, 12 C-Blocks, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 40% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% +-Brightness, 20% +- 3 pixel shift, 20% 0.8-1.2 contrast augmentation.

### Change 47 — cifar10 — 2026-10-29

**Change:** 90 epochs and 7e-4 lr

**Why:** I think a gradual descent over a longer period will result in higher accuracy

**Expected:** longer training time and higher accuracy

**Result:** Validation accuracy 75.72%; train accuracy 100.00%; validation loss 1.4378; train loss 0.0001; 16.90 min on cuda. Train/validation accuracy gap: 24.28%.

**Loss curve:** Epoch loss 1.5710 → 0.0424 over 90 logged epochs.

**Interpretation / next step:** This is what I expected. Need to reduce training time. will try with less depth. 

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO13`, seed `0`, train count `45000`, validation count `5000`, run ID `20261029_205447_baseline_SiLU_WD_CAS_DA_8x8_DropO13`. SiLU , 12 C-Blocks, 90 epochs, 7e-4 lr, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% +-Brightness, 20% +- 3 pixel shift, 20% 0.8-1.2 contrast augmentation.

### Change 48 — cifar10 — 2026-10-29

**Change:** 12 --> 8 composite layers

**Why:** Need to reduce training time

**Expected:** similar accuracy and less training time

**Result:** Validation accuracy 74.44%; train accuracy 100.00%; validation loss 1.3526; train loss 0.0002; 13.86 min on cuda. Train/validation accuracy gap: 25.56%.

**Loss curve:** Epoch loss 1.5969 → 0.0508 over 90 logged epochs.

**Interpretation / next step:** Bit too much of an accuracy drop, will revert and use less augmentation %

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO14`, seed `0`, train count `45000`, validation count `5000`, run ID `20261029_211902_baseline_SiLU_WD_CAS_DA_8x8_DropO14`. SiLU , 8 C-Blocks, 90 epochs, 7e-4 lr, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% +-Brightness, 20% +- 3 pixel shift, 20% 0.8-1.2 contrast augmentation.

### Change 49 — cifar10 — 2026-10-29

**Change:** back to 12, reduced down to 15% on augmentations

**Why:** Need to reduce training time

**Expected:** similar accuracy and less training time

**Result:** Validation accuracy 74.70%; train accuracy 100.00%; validation loss 1.4290; train loss 0.0001; 15.36 min on cuda. Train/validation accuracy gap: 25.30%.

**Loss curve:** Epoch loss 1.6130 → 0.0402 over 100 logged epochs.

**Interpretation / next step:** This did not help. will revert and try out batch norm

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO15`, seed `0`, train count `45000`, validation count `5000`, run ID `20261029_213712_baseline_SiLU_WD_CAS_DA_8x8_DropO15`. SiLU , 12 C-Blocks, 90 epochs, 7e-4 lr, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 15% H-Flipping, 15% erasure, 15% +- 1- rotations, 15% +-Brightness, 15% +- 3 pixel shift, 15% 0.8-1.2 contrast augmentation.

### Change 50 — cifar10 — 2026-10-29

**Change:** back to 20%, swapped to batchNorm

**Why:** Need to increase accuracy

**Expected:** similar accuracy or better

**Result:** Validation accuracy 74.98%; train accuracy 100.00%; validation loss 1.3930; train loss 0.0001; 17.69 min on cuda. Train/validation accuracy gap: 25.02%.

**Loss curve:** Epoch loss 1.5021 → 0.0386 over 90 logged epochs.

**Interpretation / next step:** This also did not help. Will try data augmentation options

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO16`, seed `0`, train count `45000`, validation count `5000`, run ID `20261029_220833_baseline_SiLU_WD_CAS_DA_8x8_DropO16`. SiLU , 12 C-Blocks, 90 epochs, 7e-4 lr, batch norm, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% +-Brightness, 20% +- 3 pixel shift, 20% 0.8-1.2 contrast augmentation.



### Change 51 — cifar10 — 2026-10-29

**Change:** reworked data augmentation. now on the fly and 4 options (no brightness, contrast, shift)

**Why:** Need to increase accuracy

**Expected:** similar

**Result:** Validation accuracy 72.00%; train accuracy 100.00%; validation loss 1.4488; train loss 0.0007; 12.95 min on cuda. Train/validation accuracy gap: 28.00%.

**Loss curve:** Epoch loss 1.7798 → 0.0550 over 90 logged epochs.

**Interpretation / next step:** Accuracy is worse, but the training time reduction is nice. will try adjusting model size.

**Reproducibility:** Package `SiLU_WD_CAS_DA_8x8_DropO17`, seed `0`, train count `45000`, validation count `5000`, run ID `20261029_225213_baseline_SiLU_WD_CAS_DA_8x8_DropO17`.  SiLU , 12 C-Blocks, 90 epochs, 7e-4 lr, WD, patching (8x8), AdamW, 512 hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% 0.8 - 1.2


### Change 52 — cifar10 — 2026-10-30

**Change:** 4x4 and 128 hidden and 64 patch hidden and 10 depth. (big rework sorry)

**Why:** Looked to MLP-Mixer and realized i was compressing patch information instead of expanding. 

**Expected:** shorter training time

**Result:** Validation accuracy 70.04%; train accuracy 99.02%; validation loss 1.1780; train loss 0.0655; 12.43 min on cuda. Train/validation accuracy gap: 28.98%.

**Loss curve:** Epoch loss 1.7868 → 0.2439 over 90 logged epochs.

**Interpretation / next step:** Large drop in accuracy. will try 256 hidden

**Reproducibility:** Package `SiLU_WD_CAS_DA_4x4_DropO`, seed `0`, train count `45000`, validation count `5000`, run ID `20261030_134141_baseline_SiLU_WD_CAS_DA_4x4_DropO`. 
SiLU , 10 C-Blocks, 90 epochs, 7e-4 lr, WD, patching (4x4), AdamW, 128 hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% 0.8 - 1.2

### Change 53 — cifar10 — 2026-10-30

**Change:**256 hidden

**Why:** Looked to MLP-Mixer and realized i was compressing patch information instead of expanding

**Expected:** shorter training time atleast

**Result:** Validation accuracy 69.42%; train accuracy 99.98%; validation loss 1.3804; train loss 0.0123; 12.62 min on cuda. Train/validation accuracy gap: 30.56%.

**Loss curve:** Epoch loss 1.7633 → 0.1345 over 90 logged epochs.

**Interpretation / next step:** This did not work, will increase representation. 

**Reproducibility:** Package `SiLU_WD_CAS_DA_4x4_DropO2`, seed `0`, train count `45000`, validation count `5000`, run ID `20261030_135442_baseline_SiLU_WD_CAS_DA_4x4_DropO2`. SiLU , 10 C-Blocks, 90 epochs, 7e-4 lr, WD, patching (4x4), AdamW, 256 hidden, 64 patch hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% 0.8 - 1.2

### Change 54 — cifar10 — 2026-10-30

**Change:** 4x4 and 512 hidden

**Why:** Hoping to expand representation in mlp block

**Expected:** higher accuracy

**Result:** Validation accuracy 71.42%; train accuracy 99.99%; validation loss 1.3652; train loss 0.0060; 13.78 min on cuda. Train/validation accuracy gap: 28.57%.

**Loss curve:** Epoch loss 1.7726 → 0.1062 over 90 logged epochs.

**Interpretation / next step:** This was a slight improvement.

**Reproducibility:** Package `SiLU_WD_CAS_DA_4x4_DropO3`, seed `0`, train count `45000`, validation count `5000`, run ID `20261030_141729_baseline_SiLU_WD_CAS_DA_4x4_DropO3`. SiLU , 1 C-Blocks, 90 epochs, 7e-4 lr, WD, patching (4x4), AdamW, 512 hidden, 64 patch hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% erasure, 20% +- 1- rotations, 20% 0.8 - 1.2

### Change 55 — cifar10 — 2026-10-30

**Change:** Added cutmix and removed erasure

**Why:** Added cutmix because it worked for MLP mixer

**Expected:** Higher loss and better generalization

**Result:** Validation accuracy 75.08%; train accuracy 95.07%; validation loss 0.7664; train loss 0.2187; 13.17 min on cuda. Train/validation accuracy gap: 19.99%.

**Loss curve:** Epoch loss 1.9713 → 0.8860 over 90 logged epochs.

**Interpretation / next step:** This was a crazy improvement. Will keep and lower batch size to get more updates.

**Reproducibility:** Package `SiLU_WD_CAS_DA_4x4_DropO4`, seed `0`, train count `45000`, validation count `5000`, run ID `20261030_155143_baseline_SiLU_WD_CAS_DA_4x4_DropO4`. SiLU , 10 C-Blocks, 90 epochs, 7e-4 lr, WD, patching (4x4), AdamW, 512 hidden, 64 patch hidden, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% cut mix, 20% +- 1- rotations, 20% 0.8 - 1.2

### Change 56 — cifar10 — 2026-10-30

**Change:** 256 --> 128 batch size

**Why:** More updates might mean better late accuracy

**Expected:** Higher accuracy and longer training time

**Result:** Validation accuracy 77.00%; train accuracy 98.76%; validation loss 0.7404; train loss 0.1036; 14.42 min on cuda. Train/validation accuracy gap: 21.76%.

**Loss curve:** Epoch loss 1.9329 → 0.7981 over 90 logged epochs.

**Interpretation / next step:**  good jump in accuracy. will try adjusting lr to squeeze out more early improvement

**Reproducibility:** Package `SiLU_WD_CAS_DA_4x4_DropO5`, seed `0`, train count `45000`, validation count `5000`, run ID `20261030_161037_baseline_SiLU_WD_CAS_DA_4x4_DropO5`. SiLU , 10 C-Blocks, 90 epochs, 7e-4 lr, WD, patching (4x4), AdamW, 512 hidden, 64 patch hidden, 128 batch size, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% cut mix, 20% +- 1- rotations, 20% 0.8 - 1.2

### Change 57 — cifar10 — 2026-10-30

**Change:** 7e-4 --> 8e-4 lr

**Why:** reverting on past experience and hoping that more aggresive learning rate may help

**Expected:** Higher accuracy

**Result:** Validation accuracy 78.24%; train accuracy 99.39%; validation loss 0.7067; train loss 0.0739; 16.71 min on cuda. Train/validation accuracy gap: 21.15%.

**Loss curve:** Epoch loss 1.9345 → 0.7567 over 90 logged epochs.

**Interpretation / next step:** Good improvement. will increase further

**Reproducibility:** Package `SiLU_WD_CAS_DA_4x4_DropO6`, seed `0`, train count `45000`, validation count `5000`, run ID `20261030_164922_baseline_SiLU_WD_CAS_DA_4x4_DropO6`. SiLU , 10 C-Blocks, 90 epochs, 8e-4 lr, WD, patching (4x4), AdamW, 512 hidden, 64 patch hidden, 128 batch size, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% cut mix, 20% +- 1- rotations, 20% 0.8 - 1.2

### Change 58 — cifar10 — 2026-10-30

**Change:** 8e-4 --> 1e-3 lr

**Why:** reverting on past experience and hoping that more aggresive learning rate may help

**Expected:** Higher accuracy

**Result:** Validation accuracy 80.52%; train accuracy 99.81%; validation loss 0.6681; train loss 0.0461; 16.30 min on cuda. Train/validation accuracy gap: 19.29%.

**Loss curve:** Epoch loss 1.9325 → 0.7024 over 90 logged epochs.

**Interpretation / next step:** This was a big improvement. Will call it here.

**Reproducibility:** Package `SiLU_WD_CAS_DA_4x4_DropO7`, seed `0`, train count `45000`, validation count `5000`, run ID `20261030_171014_baseline_SiLU_WD_CAS_DA_4x4_DropO7`. SiLU , 10 C-Blocks, 90 epochs, 1e-3 lr, WD, patching (4x4), AdamW, 512 hidden, 64 patch hidden, 128 batch size, cosine annealing scheduler, 20% dropout, 20% H-Flipping, 20% cut mix, 20% +- 1- rotations, 20% 0.8 - 1.2

