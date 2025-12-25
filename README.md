# Credit Card Fraud Detection (Imbalanced Classification)

신용카드 거래 데이터에서 사기 거래(Class=1)를 탐지하는 이진 분류 모델을 구현했습니다.  
데이터의 극단적 클래스 불균형(사기 비율 약 0.17%) 문제를 고려하여  
**cost-sensitive learning**과 **PR-AUC 기반 평가**를 적용했습니다.

본 프로젝트는 단순 모델 구현이 아니라,  
불균형 데이터 환경에서의 **평가 지표 선택**, **실험 설계**, **하이퍼파라미터 최적화**에
초점을 두고 진행되었습니다.

---

## Dataset
- Kaggle Credit Card Fraud Dataset  
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

> ⚠️ 데이터 파일(`creditcard.csv`)은 레포에 포함되어 있지 않습니다.  
> 위 링크에서 데이터를 다운로드한 후, 프로젝트 루트 디렉토리에  
> `creditcard.csv` 이름으로 저장해 주세요.

---

## Problem Setting
- Binary classification (정상 거래 vs 사기 거래)
- Extreme class imbalance (fraud ratio ≈ 0.17%)
- Accuracy는 의미 있는 지표가 아니므로 **PR-AUC**를 주요 평가 지표로 사용

---

## Methods

### Baseline Model
- **Logistic Regression**
- `class_weight="balanced"`를 사용하여 비용 민감 학습(cost-sensitive learning) 적용

### Tree-based Model
- **XGBoost (Gradient Boosted Decision Trees)**
- 클래스 불균형을 반영하기 위해 `scale_pos_weight` 사용
- PR-AUC를 목표로 Optuna를 이용한 하이퍼파라미터 튜닝 수행

---

## Hyperparameter Tuning
- Optimizer: **Optuna (Bayesian Optimization)**
- Objective metric: **PR-AUC (Average Precision)**
- **Stratified K-Fold Cross-Validation (K=5)** 기반으로  
  각 fold의 PR-AUC 평균을 최대화하도록 탐색
- `scale_pos_weight`는 이론값(negative/positive 비율) 주변에서 탐색

---

## Preprocessing
- 입력 특성: `V1 ~ V28` + `Amount`
- `V1 ~ V28`은 이미 변환된 특성(PCA 기반)으로 가정
- `Amount` 특성만 train set 기준으로 `StandardScaler` 적용
- Validation/Test set에는 train 기준으로 학습된 scaler만 사용하여 **data leakage 방지**

---

## Evaluation
- Train / Validation / Test = 60 / 20 / 20 분할
- 하이퍼파라미터 튜닝은 train+validation 데이터로만 수행
- 최종 성능 평가는 **held-out test set의 PR-AUC**로 측정
- 확률 기반 예측(`predict_proba`)을 사용하여 threshold-independent 평가 수행

---

## Output
- Test set에 대한 사기 거래 확률 예측 결과를 Score/score.csv 파일로 저장합니다.

---

## Key Takeaways
- 극단적 불균형 데이터에서는 Accuracy나 ROC-AUC보다 PR-AUC가 더 적절한 평가 지표
- Cost-sensitive learning은 소수 클래스 탐지 성능을 크게 개선함
- Stratified K-Fold CV 기반 하이퍼파라미터 튜닝은 단일 validation split 대비 안정적인 성능 추정 제공
