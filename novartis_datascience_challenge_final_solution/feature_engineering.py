import  pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import recall_score
import pickle
import os
import category_encoders as ce
import numpy as np

NUM_OF_BOOST_ROUND = 1000
EARLY_STOPPING = 100

def perform_chi2_test(df, column, target):
    data_crosstab = pd.crosstab(df[column],
                                df[target],
                                margins=False)
    stat, p, dof, expected = chi2_contingency(data_crosstab)
    prob = 0.95
    # interpret p-value
    alpha = 1.0 - prob
    print('%s dof=%d, significance=%.3f, p_value=%.3f' % (column, dof, alpha, p))

    if p <= alpha:
        # print('Dependent (reject H0)')
        return True , round(alpha,3) , round(p,3)
    else:
        # print('Independent (fail to reject H0)')
        return False, round(alpha,3) , round(p,3)

def create_features(X_train, X_test, remove_non_significant_features=False):
    # Extract features from date

    X_train['day'] = X_train.DATE.apply(lambda x: x.day)
    X_train['month'] = X_train.DATE.apply(lambda x: x.month)
    X_train['year'] = X_train.DATE.apply(lambda x: x.year)
    X_train['dayofweek'] = X_train.DATE.apply(lambda x: x.dayofweek)

    X_test['day'] = X_test.DATE.apply(lambda x: x.day)
    X_test['month'] = X_test.DATE.apply(lambda x: x.month)
    X_test['year'] = X_test.DATE.apply(lambda x: x.year)
    X_test['dayofweek'] = X_test.DATE.apply(lambda x: x.dayofweek)


    X = X_train.drop(['INCIDENT_ID', 'DATE', 'MULTIPLE_OFFENSE'], axis=1)
    y = X_train.MULTIPLE_OFFENSE

    XTEST = X_test.drop(['INCIDENT_ID', 'DATE'], axis=1)

    if remove_non_significant_features:
        features_to_remove = []
        for col in X.columns:
            isSignificant, alpha, p = perform_chi2_test(X_train, col, 'MULTIPLE_OFFENSE')
            if isSignificant == False:
                features_to_remove.append(col)

        X = X.drop(features_to_remove, axis=1)
        XTEST = XTEST.drop(features_to_remove, axis=1)


    high_cardinality_features = X.columns
    oof = pd.DataFrame([])
    smoothing = 0.10

    for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=2020, shuffle=True).split(X, y):
        ce_target_encoder = ce.TargetEncoder(cols=high_cardinality_features, smoothing=smoothing, min_samples_leaf=10)
        ce_target_encoder.fit(X.iloc[tr_idx, :], y[tr_idx])
        oof = oof.append(ce_target_encoder.transform(X.iloc[oof_idx, :]), ignore_index=False)

    ce_target_encoder = ce.TargetEncoder(cols=high_cardinality_features, smoothing=smoothing)
    ce_target_encoder.fit(X, y)
    test = ce_target_encoder.transform(XTEST)

    oof = oof.sort_index()
    X_te_features = oof
    XTest_te_features = test

    for col in X_te_features.columns:
        X['te_' + col] = X_te_features[col]
        XTEST['te_' + col] = XTest_te_features[col]

    return X, y, XTEST

def RECALL(preds, train_data):
    labels = train_data.get_label()
    _recall = recall_score( labels, (preds >= 0.5), average='macro')
    return 'recall', _recall, True

def get_evals_dataframe(evals_result):
    train_logloss = evals_result['training']['binary_logloss']
    valid_logloss = evals_result['valid_1']['binary_logloss']

    train_recall = evals_result['training']['recall']
    valid_recall = evals_result['valid_1']['recall']
    iterations = list(range(1, len(train_logloss) + 1))

    evals_result_df = pd.DataFrame()
    evals_result_df['recall_score'] = train_recall + valid_recall
    evals_result_df['Recall'] = ['Train'] * len(train_recall) + ['Valid'] * len(valid_recall)
    evals_result_df['iterations'] = iterations + iterations

    evals_result_df['binarylogloss_score'] = train_logloss + valid_logloss
    evals_result_df['Binary Logloss'] = ['Train'] * len(train_recall) + ['Valid'] * len(valid_recall)
    return evals_result_df

def get_featureImportance(model):
    features = model.feature_name()
    importance = list(model.feature_importance())

    tmp = pd.DataFrame(index=features)
    tmp['Feature Importance'] = importance
    tmp = tmp.sort_values(by='Feature Importance', ascending=False)
    return tmp


def get_evals_dataframe_xgb(evals_result):
    train_logloss = evals_result['train']['error']
    valid_logloss = evals_result['eval']['error']

    train_recall = [ abs(_) for _ in evals_result['train']['recall']]
    valid_recall = [ abs(_) for _ in evals_result['eval']['recall']]

    iterations = list(range(1, len(train_logloss) + 1))

    evals_result_df = pd.DataFrame()
    evals_result_df['recall_score'] = train_recall + valid_recall
    evals_result_df['Recall'] = ['Train'] * len(train_recall) + ['Valid'] * len(valid_recall)
    evals_result_df['iterations'] = iterations + iterations

    evals_result_df['binarylogloss_score'] = train_logloss + valid_logloss
    evals_result_df['Binary Logloss'] = ['Train'] * len(train_recall) + ['Valid'] * len(valid_recall)
    return evals_result_df

def get_featureImportance_xgb(model):
    fp_info = model.get_score(importance_type="gain")
    features = fp_info.keys()
    importance = fp_info.values()

    tmp = pd.DataFrame(index=features)
    tmp['Feature Importance'] = importance
    tmp = tmp.sort_values(by='Feature Importance', ascending=False)
    return tmp





benchmark_model_information = './benchmark_model_information.pkl'
final_customized_model_information = "./final_customized_model_information.pkl"

def benchmark_model(df, features,target):
    SEED = 1
    from numpy.random import seed
    seed(SEED)
    import os
    os.environ['PYTHONHASHSEED'] = str(SEED)

    if os.path.exists(benchmark_model_information):
        print("Previous saved information")
        with open(benchmark_model_information, 'rb') as f:
            benchmark_info = pickle.load(f)
            evals_result_xgb_df = get_evals_dataframe_xgb(benchmark_info['evals_result'])
            validation_score_xgb = benchmark_info['validation_score']

    else:
        benchmark_info = {}

        X = df[features]
        y = df[target]

        X_TRAIN, X_VALID, y_train, y_valid = train_test_split(X,y, test_size=0.2, random_state=1, stratify=y)

    # XGB
        dtrain = xgb.DMatrix(X_TRAIN, label=y_train)
        dvalid = xgb.DMatrix(X_VALID, label=y_valid)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        xgb_param = {'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic', 'seed':1}
        xgb_param['nthread'] = 4
        evals_result_xgb = {}
        num_round = NUM_OF_BOOST_ROUND
        bst_xgb = xgb.train(xgb_param, dtrain, num_round, evals=watchlist, verbose_eval=50,
                            evals_result=evals_result_xgb,
                            early_stopping_rounds=EARLY_STOPPING, feval=RECALL_XGB)

        tmp_pred = bst_xgb.predict(dvalid, ntree_limit=bst_xgb.best_iteration + 1)
        tmp_pred = (tmp_pred >= 0.5).astype('int')
        validation_score_xgb = round(recall_score(y_valid, tmp_pred, average='macro'), 4)
        print("\n\nValidation_score_xgb :", validation_score_xgb)
        evals_result_xgb_df = get_evals_dataframe_xgb(evals_result_xgb)

        with open(benchmark_model_information, 'wb') as f:
            benchmark_info['model'] = bst_xgb
            benchmark_info['validation_score'] = validation_score_xgb
            benchmark_info['evals_result'] = evals_result_xgb

            pickle.dump(benchmark_info, f)

    print(evals_result_xgb_df.head())

    return evals_result_xgb_df, get_featureImportance_xgb(benchmark_info['model']), validation_score_xgb

def RECALL_XGB(preds, train_data):
    labels = train_data.get_label()
    _recall = recall_score( labels, (preds >= 0.5), average='macro')
    return 'recall', _recall * -1

def final_customized_model(train_df, test_df):
    SEED = 1
    from numpy.random import seed
    seed(SEED)
    import os
    os.environ['PYTHONHASHSEED'] = str(SEED)

    if os.path.exists(final_customized_model_information):
        print("Previous saved information")
        with open(final_customized_model_information, 'rb') as f:
            benchmark_info = pickle.load(f)
            evals_result_xgb_df = get_evals_dataframe_xgb(benchmark_info['evals_result'])
            validation_score_xgb = benchmark_info['validation_score']

    else:

        X, y, XTEST = create_features(train_df, test_df, True)
        TRAIN_ON_FULL_DATA = True
        benchmark_info = {}
        best_hyperparameters = {'bagging_fraction': 0.6070788288480509,
                              'bagging_freq': 5,
                              'eta': 0.325,
                              'feature_fraction': 0.8492952923651169,
                              'lambda_l1': 1.0,
                              'lambda_l2': 2.85,
                              'max_depth': 66,
                              'min_data_in_leaf': 19,
                              'min_sum_hessian_in_leaf': 1.4019779348104815,
                              'num_leaves': 63,
                              'objective': 'binary',
                              'scale_pos_weight': 2.0500000000000003,
                              'seed': 1,
                              'n_estimators': 17
                            }

        X_TRAIN, X_VALID, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

        # XGB

        dtrain = xgb.DMatrix(X_TRAIN, label=y_train)
        dvalid = xgb.DMatrix(X_VALID, label=y_valid)

        best_hyperparameters_xgb = {'booster': 'gbtree',
                                    'colsample_bytree': 0.9,
                                    'eta': 0.5,
                                    'gamma': 0.9,
                                    'max_depth': 3,
                                    'min_child_weight': 1.0,
                                    'n_estimators': 59.0,
                                    'nthread': 4,
                                    'objective': 'binary:logistic',
                                    'reg_alpha': 0.05,
                                    'reg_lambda': 1.6500000000000001,
                                    'scale_pos_weight': 3.7,
                                    'seed': 1,
                                    'subsample': 0.9500000000000001,
                                    'tree_method': 'exact',
                                    'n_estimators': 5}

        xgb_param = best_hyperparameters_xgb.copy()
        del xgb_param['n_estimators']

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        evals_result_xgb = {}
        num_round = NUM_OF_BOOST_ROUND
        bst_xgb = xgb.train(xgb_param, dtrain, num_round, evals=watchlist, verbose_eval=50,
                            evals_result=evals_result_xgb,
                            early_stopping_rounds=EARLY_STOPPING, feval=RECALL_XGB)

        tmp_pred = bst_xgb.predict(dvalid, ntree_limit=bst_xgb.best_iteration + 1)
        tmp_pred = (tmp_pred >= 0.5).astype('int')
        validation_score_xgb = round(recall_score(y_valid, tmp_pred, average='macro'), 4)
        print("\n\nValidation_score_xgb :", validation_score_xgb)
        evals_result_xgb_df = get_evals_dataframe_xgb(evals_result_xgb)


        if TRAIN_ON_FULL_DATA:
            best_param = best_hyperparameters_xgb.copy()
            num_round = best_param['n_estimators']

            dtrain_full = xgb.DMatrix(X, label=y)

            model = xgb.train(best_param,
                              dtrain_full,
                              num_round,
                              )

            xgb_full_pred_test_prob = model.predict(xgb.DMatrix(XTEST), ntree_limit=model.best_iteration + 1)
            print(xgb_full_pred_test_prob.shape)

            print("Saving submission file: training on full_data using best_params")
            xgb_best_param_with_full_data_training = test_df[['INCIDENT_ID']].copy()

            final_prediction = (xgb_full_pred_test_prob >= 0.5).astype('int')

            xgb_best_param_with_full_data_training['MULTIPLE_OFFENSE'] = final_prediction

            xgb_best_param_with_full_data_training.to_csv("xgb_best_param_with_full_data_training.csv", index=False)


        with open(final_customized_model_information, 'wb') as f:
            benchmark_info['model'] = bst_xgb
            benchmark_info['validation_score'] = validation_score_xgb
            benchmark_info['evals_result'] = evals_result_xgb
            pickle.dump(benchmark_info, f)

    return evals_result_xgb_df, get_featureImportance_xgb(benchmark_info['model']), validation_score_xgb
