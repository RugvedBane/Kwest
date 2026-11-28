import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

BASE = '/kaggle/input/competitions/playground-series-s6e11'
train = pd.read_csv(f'{BASE}/train.csv')
test  = pd.read_csv(f'{BASE}/test.csv')

TARGET = [c for c in train.columns if c not in test.columns][0]
print('TARGET:', TARGET)
y = train[TARGET].astype(int).values
test_ids = test['id'].values

RAW = [c for c in train.columns if c not in (TARGET, 'id')]

def build(df):
    d = pd.DataFrame(index=df.index)
    for c in RAW:
        d[c] = df[c]
    d['amt_last_digit'] = (df['amount_usd'] * 100).round().astype('int64') % 10
    d['has_disputes'] = (df['prior_disputes'] > 0).astype(int)
    d['card_age_ratio'] = df['card_age_months'] / (df['customer_age'] * 12 + 1)
    d['amt_per_balance'] = df['amount_usd'] / (df['account_balance_usd'] + 1)
    d['dist_per_hour'] = df['distance_from_home_km'] / (df['hours_since_last_txn'] + 1)
    d['cvv_per_velocity'] = df['cvv_retry_count'] / (df['velocity_score'] + 1)
    d['risk_combo'] = df['merchant_risk_score'] * df['velocity_score']
    d['risk_sum'] = df['merchant_risk_score'] + df['velocity_score']
    d['high_risk_flag'] = ((df['merchant_risk_score'] > df['merchant_risk_score'].median()) &
                           (df['velocity_score'] > df['velocity_score'].median())).astype(int)
    return d

X  = build(train)
Xt = build(test)

CATS = [c for c in X.columns if X[c].dtype == 'object' or str(X[c].dtype) == 'str']
for c in CATS:
    cats = pd.Index(sorted(set(X[c].astype(str)) | set(Xt[c].astype(str))))
    X[c]  = pd.Categorical(X[c].astype(str), categories=cats)
    Xt[c] = pd.Categorical(Xt[c].astype(str), categories=cats)
for c in X.columns:
    if X[c].dtype == bool:
        X[c] = X[c].astype(int); Xt[c] = Xt[c].astype(int)

print('n features:', X.shape[1], 'cats:', CATS)

NF = 5
skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=42)
folds = list(skf.split(X, y))

SEEDS = [42, 7]

# ---------------- LightGBM (CPU) ----------------
def run_lgb(seed):
    oof = np.zeros(len(X)); pred = np.zeros(len(Xt)); fa = []
    for i, (tr, va) in enumerate(folds):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        m = lgb.LGBMClassifier(
            n_estimators=20000, learning_rate=0.01, num_leaves=4,
            min_child_samples=40, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1,
            random_state=seed)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='auc',
              callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[va] = m.predict_proba(Xva)[:, 1]
        pred += m.predict_proba(Xt)[:, 1] / NF
        fa.append(roc_auc_score(yva, oof[va]))
        print(f'  LGB seed {seed} fold {i}: AUC {fa[-1]:.6f}  best_iter {m.best_iteration_}')
    return oof, pred, fa

# ---------------- XGBoost (CPU) ----------------
def run_xgb(seed):
    oof = np.zeros(len(X)); pred = np.zeros(len(Xt)); fa = []
    for i, (tr, va) in enumerate(folds):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        m = xgb.XGBClassifier(
            n_estimators=20000, learning_rate=0.01, max_depth=3,
            min_child_weight=40, subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, tree_method='hist',
            enable_categorical=True, eval_metric='auc',
            early_stopping_rounds=300, random_state=seed, n_jobs=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        oof[va] = m.predict_proba(Xva)[:, 1]
        pred += m.predict_proba(Xt)[:, 1] / NF
        fa.append(roc_auc_score(yva, oof[va]))
        print(f'  XGB seed {seed} fold {i}: AUC {fa[-1]:.6f}  best_iter {m.best_iteration}')
    return oof, pred, fa

# ---------------- CatBoost (CPU) ----------------
def run_cat(seed):
    oof = np.zeros(len(X)); pred = np.zeros(len(Xt)); fa = []
    for i, (tr, va) in enumerate(folds):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        m = CatBoostClassifier(
            iterations=20000, learning_rate=0.01, depth=4,
            l2_leaf_reg=1.0, subsample=0.9, rsm=0.8,
            thread_count=-1, eval_metric='AUC',
            early_stopping_rounds=300, random_seed=seed, verbose=False,
            allow_writing_files=False)
        m.fit(Xtr, ytr, eval_set=(Xva, yva), cat_features=CATS)
        oof[va] = m.predict_proba(Xva)[:, 1]
        pred += m.predict_proba(Xt)[:, 1] / NF
        fa.append(roc_auc_score(yva, oof[va]))
        print(f'  CAT seed {seed} fold {i}: AUC {fa[-1]:.6f}  best_iter {m.get_best_iteration()}')
    return oof, pred, fa

results = {}
for name, fn in [('lgb', run_lgb), ('xgb', run_xgb), ('cat', run_cat)]:
    oofs, preds, fas = [], [], []
    for s in SEEDS:
        print(f'\n=== {name} seed {s} ===')
        o, p, f = fn(s)
        oofs.append(o); preds.append(p); fas.append(f)
    oof_avg = np.mean(oofs, axis=0)
    pred_avg = np.mean(preds, axis=0)
    auc = roc_auc_score(y, oof_avg)
    results[name] = dict(oof=oof_avg, pred=pred_avg, auc=auc, fold_aucs=fas)
    print(f'\n{name.upper()} pooled OOF AUC (2-seed avg): {auc:.6f}')

# ---- rank-average blend of the three models ----
from scipy.stats import rankdata
oof_rank = np.mean([rankdata(results[k]['oof']) for k in results], axis=0)
pred_rank = np.mean([rankdata(results[k]['pred']) for k in results], axis=0)
auc_blend = roc_auc_score(y, oof_rank)

print('\n--- model comparison (pooled OOF AUC, 2-seed avg) ---')
for k in results:
    print(f'{k.upper():>4}: {results[k]["auc"]:.6f}')
print(f'BLEND (rank avg of lgb/xgb/cat): {auc_blend:.6f}')

print('\n--- per-fold AUC of blend vs best single ---')
best_single = max(results, key=lambda k: results[k]['auc'])
improved = 0
for i in range(NF):
    va = folds[i][1]
    b = roc_auc_score(y[va], results[best_single]['oof'][va])
    a = roc_auc_score(y[va], oof_rank[va])
    d = a - b
    if d > 0: improved += 1
    print(f'fold {i}: {best_single} {b:.6f}  blend {a:.6f}  delta {d:+.6f}')
print(f'Folds improved by blend: {improved}/{NF}')

pd.DataFrame({'id': test_ids, TARGET: pred_rank}).to_csv('submission_candidate.csv', index=False)
print('saved submission_candidate.csv (submission.csv untouched)')
print(f'FINAL_OOF: auc {auc_blend:.6f}')
