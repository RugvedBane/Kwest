import glob, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

t0 = time.time()
BUDGET = 5400.0   # stop adding new seeds after this many seconds

tr_path = [p for p in glob.glob('/kaggle/input/**/*.csv', recursive=True) if os.path.basename(p)=='train.csv'][0]
te_path = [p for p in glob.glob('/kaggle/input/**/*.csv', recursive=True) if os.path.basename(p)=='test.csv'][0]
train = pd.read_csv(tr_path); test = pd.read_csv(te_path)
print(train.shape, test.shape, flush=True)

TARGET = 'is_fraud'
CATS = ['merchant_category','card_type','auth_method','channel','device_type']
y = train[TARGET].astype(int).values

def fe(df):
    d = df.copy()
    amt = d['amount_usd'].astype(float).values
    cents = np.rint(amt*100).astype(np.int64)
    int_part = np.floor(amt).astype(np.int64)
    frac_cents = cents - int_part*100
    d['amt_d_units']   = int_part % 10
    d['amt_d_tens']    = (int_part//10) % 10
    d['amt_d_hunds']   = (int_part//100) % 10
    d['amt_d_thous']   = (int_part//1000) % 10
    d['amt_d_10thou']  = (int_part//10000) % 10
    d['amt_first_dig'] = np.where(int_part>0, int_part//(10**np.maximum(np.floor(np.log10(np.maximum(int_part,1))).astype(int),0)), 0)
    d['amt_ndigits']   = np.where(int_part>0, np.floor(np.log10(np.maximum(int_part,1))).astype(int)+1, 0)
    d['amt_digit_sum'] = ((int_part//1)%10)+((int_part//10)%10)+((int_part//100)%10)+((int_part//1000)%10)+((int_part//10000)%10)
    d['amt_cent_units'] = frac_cents % 10
    d['amt_cent_tens']  = (frac_cents//10) % 10
    d['amt_frac_zero']  = (frac_cents==0).astype(np.int8)
    d['amt_is_round100']= (int_part%100==0).astype(np.int8)
    d['amt_trailing9']  = ((int_part%10)==9).astype(np.int8)
    d['amt_digit_repeat'] = (d['amt_d_units']==d['amt_d_tens']).astype(np.int8)
    d['amt_log'] = np.log1p(np.abs(amt))
    d['amt_frac'] = amt - int_part
    bal = d['account_balance_usd'].astype(float).values
    d['amt_bal_ratio'] = amt/(bal+1.0)
    d['amt_bal_diff']  = amt - bal
    d['bal_per_age']   = bal/(d['customer_age']+1.0)
    d['amt_per_age']   = amt/(d['customer_age']+1.0)
    d['vel_x_risk']    = d['velocity_score']*d['merchant_risk_score']
    d['txn_per_hour']  = d['txn_count_last_24h']/(d['hours_since_last_txn']+1.0)
    d['dist_per_age']  = d['distance_from_home_km']/(d['card_age_months']+1.0)
    d['risk_flags']    = (d['is_foreign_transaction'].astype(int)+d['is_new_merchant'].astype(int)
                          +d['used_vpn'].astype(int)+d['ip_country_mismatch'].astype(int)
                          +d['billing_shipping_mismatch'].astype(int)+d['is_ai_generated_scam_attempt'].astype(int))
    d['hour_sin']=np.sin(2*np.pi*d['time_of_day_hour']/24); d['hour_cos']=np.cos(2*np.pi*d['time_of_day_hour']/24)
    d['dow_sin'] =np.sin(2*np.pi*d['day_of_week']/7);       d['dow_cos'] =np.cos(2*np.pi*d['day_of_week']/7)
    d['cvv_vel'] = d['cvv_retry_count']*d['velocity_score']
    for c in CATS: d[c]=d[c].astype('category')
    return d

train_f, test_f = fe(train), fe(test)
for c in CATS:
    cats = sorted(set(train[c].dropna().unique())|set(test[c].dropna().unique()))
    train_f[c]=train_f[c].cat.set_categories(cats); test_f[c]=test_f[c].cat.set_categories(cats)

FEAT = [c for c in train_f.columns if c not in ['id', TARGET]]
Xtr, Xte = train_f[FEAT], test_f[FEAT]
ids_te = test['id'].values
cat_idx = [FEAT.index(c) for c in CATS]
print("n features:", len(FEAT), flush=True)

def logit(p):
    p = np.clip(p, 1e-7, 1-1e-7); return np.log(p/(1-p))

MODELS = ['lgb','xgb','cat']
SEEDS  = [42, 202, 777]

oof_s = {m: {} for m in MODELS}   # oof_s[m][seed] -> array
pte_s = {m: {} for m in MODELS}

for si, sd in enumerate(SEEDS):
    if si > 0 and (time.time()-t0) > BUDGET:
        print("time budget hit -> stopping after %d seed(s)" % si, flush=True); break
    print("\n############ SEED %d (%d/%d) ############" % (sd, si+1, len(SEEDS)), flush=True)

    for m in MODELS:
        oof_s[m][sd] = np.zeros(len(Xtr)); pte_s[m][sd] = np.zeros(len(Xte))

    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=sd).split(Xtr, y))

    for k,(tri,vai) in enumerate(folds):
        ytr_, yva_ = y[tri], y[vai]

        Xt, Xv = Xtr.iloc[tri], Xtr.iloc[vai]

        m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.03, num_leaves=127,
                               min_child_samples=60, subsample=0.85, subsample_freq=1,
                               colsample_bytree=0.75, reg_alpha=0.3, reg_lambda=1.0,
                               device='gpu', random_state=sd+k, verbose=-1)
        m.fit(Xt, ytr_, eval_set=[(Xv,yva_)], eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof_s['lgb'][sd][vai] = m.predict_proba(Xv)[:,1]
        pte_s['lgb'][sd] += m.predict_proba(Xte)[:,1]/5
        a_lgb = roc_auc_score(yva_, oof_s['lgb'][sd][vai])

        m = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.03, max_depth=8,
                              min_child_weight=5, subsample=0.85, colsample_bytree=0.75,
                              reg_alpha=0.3, reg_lambda=1.0, tree_method='hist',
                              device='cuda', enable_categorical=True, max_cat_to_onehot=8,
                              eval_metric='auc', early_stopping_rounds=100, random_state=sd+k)
        m.fit(Xt, ytr_, eval_set=[(Xv,yva_)], verbose=False)
        oof_s['xgb'][sd][vai] = m.predict_proba(Xv)[:,1]
        pte_s['xgb'][sd] += m.predict_proba(Xte)[:,1]/5
        a_xgb = roc_auc_score(yva_, oof_s['xgb'][sd][vai])

        cb_t, cb_v, cb_e = Xt.copy(), Xv.copy(), Xte.copy()
        for c in CATS:
            cb_t[c]=cb_t[c].astype(str); cb_v[c]=cb_v[c].astype(str); cb_e[c]=cb_e[c].astype(str)
        m = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
                               loss_function='Logloss', eval_metric='AUC', task_type='GPU',
                               od_type='Iter', od_wait=100, random_seed=sd+k, verbose=0,
                               bootstrap_type='Bernoulli', subsample=0.85)
        m.fit(cb_t, ytr_, eval_set=(cb_v,yva_), cat_features=cat_idx, use_best_model=True)
        oof_s['cat'][sd][vai] = m.predict_proba(cb_v)[:,1]
        pte_s['cat'][sd] += m.predict_proba(cb_e)[:,1]/5
        a_cat = roc_auc_score(yva_, oof_s['cat'][sd][vai])

        print("seed %d fold %d | lgb %.6f  xgb %.6f  cat %.6f | %.0fs" %
              (sd, k+1, a_lgb, a_xgb, a_cat, time.time()-t0), flush=True)

# ---------------- per-seed and seed-averaged OOF AUC ----------------
used_seeds = list(oof_s['lgb'].keys())
print("\n---- OOF AUC by seed ----", flush=True)
for sd in used_seeds:
    for m in MODELS:
        print("  seed %d %-4s %.6f" % (sd, m, roc_auc_score(y, oof_s[m][sd])), flush=True)
print("\n---- OOF AUC seed-averaged (logit mean) ----", flush=True)
oof_avg, pte_avg = {}, {}
for m in MODELS:
    oof_avg[m] = logit(np.mean([oof_s[m][sd] for sd in used_seeds], axis=0))
    pte_avg[m] = logit(np.mean([pte_s[m][sd] for sd in used_seeds], axis=0))
    print("  %-4s %.6f" % (m, roc_auc_score(y, oof_avg[m])), flush=True)

# candidate pools
names9 = [f"{m}_s{sd}" for m in MODELS for sd in used_seeds]
P9 = np.column_stack([logit(oof_s[m][sd]) for m in MODELS for sd in used_seeds])
T9 = np.column_stack([pte_s[m][sd]              for m in MODELS for sd in used_seeds])
names3 = MODELS
P3 = np.column_stack([oof_avg[m] for m in MODELS])
T3 = np.column_stack([pte_avg[m] for m in MODELS])
P3_rank_auc = roc_auc_score(y, P3.mean(1))
print("\nsimple mean-3 OOF AUC: %.6f" % P3_rank_auc, flush=True)

# ---------------- LR meta-learner ----------------
def lr_meta(P, T, y, label):
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(P, y)
    auc = roc_auc_score(y, lr.predict_proba(P)[:,1])
    print("%s OOF AUC: %.6f | coef %s | b %.4f" %
          (label, auc, np.round(lr.coef_[0],4), lr.intercept_[0]), flush=True)
    return auc, lr.predict_proba(T)[:,1]

auc_lr3, pred_lr3 = lr_meta(P3, T3, y, "LR meta (3 seed-averaged)")
auc_lr9, pred_lr9 = lr_meta(P9, T9, y, "LR meta (9 seed-level) ")

# ---------------- hill climbing ----------------
def hill_climb(P, y, n_iter=80, verbose=True):
    cur = np.zeros(len(y)); w = np.zeros(P.shape[1]); tot = 0
    best = -1.0; hist = []
    for it in range(n_iter):
        cand_auc, cand_j = -1.0, -1
        for j in range(P.shape[1]):
            a = roc_auc_score(y, (cur*tot + P[:,j])/(tot+1))
            if a > cand_auc: cand_auc, cand_j = a, j
        if it > 0 and cand_auc <= best + 1e-7: break
        cur = (cur*tot + P[:,cand_j])/(tot+1); tot += 1; w[cand_j] += 1
        best = cand_auc; hist.append((cand_j, cand_auc))
        if verbose: print("   hc it %2d -> cand %d  auc %.6f" % (it, cand_j, cand_auc), flush=True)
    return w/w.sum(), best

print("\nhill climbing on 9 seed-level models...", flush=True)
w9, auc_hc9 = hill_climb(P9, y, verbose=False)
pred_hc9 = T9 @ w9
print("HC(9) OOF AUC: %.6f | weights %s" % (auc_hc9, dict(zip(names9, np.round(w9,3)))), flush=True)

print("hill climbing on 3 seed-averaged models...", flush=True)
w3, auc_hc3 = hill_climb(P3, y, verbose=False)
pred_hc3 = T3 @ w3
print("HC(3) OOF AUC: %.6f | weights %s" % (auc_hc3, dict(zip(names3, np.round(w3,3)))), flush=True)

# rank-space hill climb (AUC is rank based; sometimes more stable)
def to_rank(a):
    return pd.Series(a).rank(pct=True).values
P9r = np.column_stack([to_rank(P9[:,j]) for j in range(P9.shape[1])])
w9r, auc_hc9r = hill_climb(P9r, y, verbose=False)
pred_hc9r = np.column_stack([to_rank(T9[:,j]) for j in range(T9.shape[1])]) @ w9r
print("HC(9, rank space) OOF AUC: %.6f | weights %s" % (auc_hc9r, dict(zip(names9, np.round(w9r,3)))), flush=True)

# ---------------- pick best by OOF AUC ----------------
cands = {
    "mean3":        (P3_rank_auc, T3.mean(1)),
    "LR_meta_3":    (auc_lr3, pred_lr3),
    "LR_meta_9":    (auc_lr9, pred_lr9),
    "HC_9":         (auc_hc9, pred_hc9),
    "HC_3":         (auc_hc3, pred_hc3),
    "HC_9_rank":    (auc_hc9r, pred_hc9r),
}
print("\n========= CANDIDATE SUMMARY =========", flush=True)
for k_,(a_,_) in sorted(cands.items(), key=lambda x:-x[1][0]):
    print("  %-12s %.6f" % (k_, a_), flush=True)
best_name = max(cands, key=lambda k_: cands[k_][0])
best_pred = cands[best_name][1]
print("CHOSEN: %s (OOF %.6f)" % (best_name, cands[best_name][0]), flush=True)

# normalize chosen logit blend to [0,1] via rank->sigmoid-ish scaling (AUC invariant)
p = best_pred
if p.min() < 0 or p.max() > 1:
    p = 1/(1+np.exp(-p))
sub = pd.DataFrame({'id': ids_te, TARGET: p})
sub.to_csv('submission.csv', index=False)
print(sub.head()); print("rows", len(sub), "mean %.4f" % p.mean(), "min %.6f max %.6f" % (p.min(), p.max()))
print("TOTAL %.0fs" % (time.time()-t0), flush=True)
