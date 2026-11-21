import os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb
import xgboost as xgb

BASE = '/kaggle/input/competitions/playground-series-s6e11'
train = pd.read_csv(f'{BASE}/train.csv')
test = pd.read_csv(f'{BASE}/test.csv')

TARGET = 'is_fraud'
ID = 'id'
y = train[TARGET].values

# ---------- Feature engineering: 5 new features from top signals ----------
def add_features(df):
    df = df.copy()
    # 1. Combined risk: product of the three strongest signals
    df['risk_combo'] = df['cvv_retry_count'] * df['merchant_risk_score'] * df['velocity_score']
    # 2. Sum of top-3 risk signals
    df['risk_sum'] = df['cvv_retry_count'] + df['merchant_risk_score'] + df['velocity_score']
    # 3. CVV retries relative to velocity (retry pressure)
    df['cvv_per_velocity'] = df['cvv_retry_count'] / (df['velocity_score'] + 1.0)
    # 4. Merchant risk x velocity interaction
    df['risk_x_velocity'] = df['merchant_risk_score'] * df['velocity_score']
    # 5. High-risk flag: any of the top signals above threshold
    df['high_risk_flag'] = ((df['cvv_retry_count'] > 1) |
                            (df['merchant_risk_score'] > df['merchant_risk_score'].median()) |
                            (df['velocity_score'] > df['velocity_score'].median())).astype(int)
    return df

train = add_features(train)
test = add_features(test)

X = train.drop(columns=[TARGET, ID])
X_test = test.drop(columns=[ID])

cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype('category')
    X_test[c] = pd.Categorical(X_test[c], categories=X[c].cat.categories)

print('train', X.shape, 'test', X_test.shape, 'cats', cat_cols)

NFOLD = 5
skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=42)
folds = list(skf.split(X, y))

lgb_params = dict(
    objective='binary', metric='auc', learning_rate=0.03,
    num_leaves=64, min_child_samples=100, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    n_jobs=-1, device='cpu', verbose=-1, seed=42,
)
xgb_params = dict(
    objective='binary:logistic', eval_metric='auc', learning_rate=0.03,
    max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, tree_method='hist', n_jobs=-1, seed=42,
)

oof_lgb = np.zeros(len(X)); oof_xgb = np.zeros(len(X))
pred_lgb = np.zeros(len(X_test)); pred_xgb = np.zeros(len(X_test))

for i, (tr, va) in enumerate(folds):
    Xtr, Xva = X.iloc[tr], X.iloc[va]
    ytr, yva = y[tr], y[va]

    dtr = lgb.Dataset(Xtr, ytr)
    dva = lgb.Dataset(Xva, yva)
    m = lgb.train(lgb_params, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[va] = m.predict(Xva, num_iteration=m.best_iteration)
    pred_lgb += m.predict(X_test, num_iteration=m.best_iteration) / NFOLD

    dtrx = xgb.DMatrix(Xtr, ytr, enable_categorical=True)
    dvax = xgb.DMatrix(Xva, yva, enable_categorical=True)
    dtex = xgb.DMatrix(X_test, enable_categorical=True)
    bst = xgb.train(xgb_params, dtrx, num_boost_round=3000,
                    evals=[(dvax, 'va')], early_stopping_rounds=100, verbose_eval=False)
    oof_xgb[va] = bst.predict(dvax, iteration_range=(0, bst.best_iteration + 1))
    pred_xgb += bst.predict(dtex, iteration_range=(0, bst.best_iteration + 1)) / NFOLD

    print(f'fold {i}: lgb {roc_auc_score(yva, oof_lgb[va]):.6f}  xgb {roc_auc_score(yva, oof_xgb[va]):.6f}')

auc_lgb = roc_auc_score(y, oof_lgb)
auc_xgb = roc_auc_score(y, oof_xgb)
print(f'OOF LGB: {auc_lgb:.6f}  OOF XGB: {auc_xgb:.6f}')

r_lgb = rankdata(oof_lgb) / len(oof_lgb)
r_xgb = rankdata(oof_xgb) / len(oof_xgb)
best_w, best_auc = 0.5, 0
for w in np.linspace(0, 1, 21):
    a = roc_auc_score(y, w * r_lgb + (1 - w) * r_xgb)
    if a > best_auc:
        best_auc, best_w = a, w
print(f'Best rank blend: w_lgb={best_w:.2f}  OOF AUC={best_auc:.6f}')

rt_lgb = rankdata(pred_lgb) / len(pred_lgb)
rt_xgb = rankdata(pred_xgb) / len(pred_xgb)
final = best_w * rt_lgb + (1 - best_w) * rt_xgb

sub = pd.DataFrame({ID: test[ID], TARGET: final})
sub.to_csv('submission.csv', index=False)
print('saved submission.csv', sub.shape)
print(sub.head())
