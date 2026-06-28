# Giai thich sieu tham so

## TimeSeriesSplit
- `n_splits=5`: chia train thanh 5 fold theo thoi gian.
- `gap=5`: bo qua 5 mau giua train/validation de tranh leakage tu nhan 5 phien.

## Logistic Regression
- `C`: do manh regularization (C nho = regularization manh hon).
- `solver`: thuat toan toi uu (`lbfgs`, `liblinear`).

## Random Forest
- `n_estimators`: so cay trong rung.
- `max_depth`: do sau toi da moi cay.
- `min_samples_leaf`: so mau toi thieu o nut la.
- `max_features`: so dac trung xet moi lan split.

## Gradient Boosting
- `n_estimators`: so boosting stages.
- `learning_rate`: buoc hoc moi stage.
- `max_depth`: do sau cay co so.
- `subsample`: ty le mau dung moi stage.

## Tieu chi chon model
1. OOF F1_UP @ threshold cao nhat tren tap train (out-of-fold).
2. Neu gan bang (trong margin): OOF MCC @ threshold cao hon.
3. Tie-break tiep: OOF PR-AUC cao hon.
4. Neu van gan bang: model don gian hon (LogReg > RF > GB).
5. Test set chi dung de bao cao 1 lan, khong dung de chon model.
