# Giai thich sieu tham so

## TimeSeriesSplit
- `n_splits=4`: chia train thanh 4 fold theo thoi gian tu 2021.
- `gap=5`: bo dung 5 ngay giao dich chung truoc validation; khong dem row.
- Purge: loai train row co `label_end_date >= validation_start`.
- Threshold: chon rieng tung model tu OOF TRAIN, gioi han ty le du bao UP va precision.

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
1. F1_UP tren VALIDATION cao nhat.
2. Neu bang nhau: Recall_UP cao hon.
3. Neu van bang nhau: model don gian hon (LogReg > RF > GB).
4. Refit winner tren TRAIN+VALIDATION, sau do danh gia FINAL TEST mot lan.
