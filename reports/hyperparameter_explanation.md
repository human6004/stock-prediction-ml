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
1. F1_UP tren test cao nhat.
2. Neu gan bang: Recall_UP cao hon.
3. Neu van gan bang: model don gian hon (LogReg > RF > GB).
