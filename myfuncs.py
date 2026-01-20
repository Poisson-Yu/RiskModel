import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline

import hashlib
from optbinning import OptimalBinning

import statsmodels.api as sm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score


#md5编码
def md5_code(input_data):
    '''
    input_data: str格式
    '''
    md = hashlib.md5(input_data.encode('utf-8'))
    output_code = md.hexdigest()
    return output_code




#欠采样
def random_undersampler(df, target, rate=0.1, stratify_fea=None, replace=False):
    '''
    ---负样本下采样函数---
    df: 训练样本数据, 数据框格式
    rate: 负样本与正样本的比值
    target: 样本标签列
    replace: replace=False表示无放回抽样，反之表示又放回抽样
    '''
    pos_df = df[df[target]==1]
    neg_df = df[df[target]==0]
    N = pos_df.shape[0]
    num_neg = int(N/rate)-N
    if stratify_fea is None:
        neg_df_sample = neg_df.sample(n=num_neg, replace=replace)
    else:
        neg_df_sample = pd.DataFrame()
        cats = df[stratify_fea].unique()
        for cat in cats:
            cat_N = pos_df[pos_df[stratify_fea]==cat].shape[0]
            cat_num_neg = int(cat_N/rate)-cat_N
            cat_neg_df_sample = neg_df[neg_df[stratify_fea]==cat].sample(n=cat_num_neg, replace=replace)
            neg_df_sample = pd.concat([neg_df_sample, cat_neg_df_sample])
        
    df_sample = pd.concat([pos_df, neg_df_sample])
    return df_sample





#计算PSI
def calculate_psi(base_df, test_df, bins=10, min_sample=10, special_list=[], numerical=True, out_bins=False):
    '''
    base_df: 基期数据，Series格式
    test_df：观测期数据，Series格式
    bins：连续型变量分箱数量，等频分箱；也可直接传入分箱节点列表
    min_sample：每项数据中最小样本量
    numerical：计算PSI的数据是否为连续型变量，是连续型为True, 类别型False
    '''
    if base_df.shape[0]==0 or test_df.shape[0]==0:
        print('error!!!')
        psi = np.nan
        stat_df = None
    else:
        # base_notnull_cnt = len(list(base_df.dropna()))
        # test_notnull_cnt = len(list(test_df.dropna()))

        base_null_cnt = base_df.isnull().sum()
        test_null_cnt = test_df.isnull().sum()

        base_special_cnt = base_df.isin(special_list).sum()
        test_special_cnt = test_df.isin(special_list).sum()

        base_normal_cnt = len(base_df) - base_null_cnt - base_special_cnt
        test_normal_cnt = len(test_df) - test_null_cnt - test_special_cnt

        base_normal_df = base_df[~base_df.isin(special_list)]
        test_normal_df = test_df[~test_df.isin(special_list)]

        if numerical:
            q_list = []
            if type(bins) == int:
                bin_num = min(bins, int(base_normal_cnt/min_sample))
                q_list = [x / bin_num for x in range(1, bin_num)]
                break_list = []
                for q in q_list:
                    bk = base_normal_df.quantile(q)
                    break_list.append(bk)
                break_list = sorted(list(set(break_list)))
                score_bin_list = [-np.inf] + break_list + [np.inf]
            else:
                score_bin_list = bins   #可传入分箱节点

            base_cnt_list = [base_null_cnt, base_special_cnt]
            test_cnt_list = [test_null_cnt, test_special_cnt]
            bucket_list = ['MISSING', 'Special']
            for i in range(len(score_bin_list)-1):
                left = round(score_bin_list[i+0], 4)
                right = round(score_bin_list[i+1], 4)
                bucket_list.append('(' + str(left) + ',' + str(right) + ']')

                base_cnt = base_normal_df[(base_normal_df>left) & (base_normal_df<=right)].shape[0]
                base_cnt_list.append(base_cnt)

                test_cnt = test_normal_df[(test_normal_df>left) & (test_normal_df<=right)].shape[0]
                test_cnt_list.append(test_cnt)

            stat_df = pd.DataFrame({'bucket': bucket_list, 'base_cnt': base_cnt_list, 'test_cnt': test_cnt_list})
            stat_df['base_dist'] = stat_df['base_cnt']/len(base_df)
            stat_df['test_dist'] = stat_df['test_cnt']/len(test_df)
        else:
            base_cnt_list = [base_null_cnt]
            test_cnt_list = [test_null_cnt]
            bucket_list = ['MISSING']
            stat_df = pd.DataFrame({'bucket': bucket_list, 'base_cnt': base_cnt_list, 'test_cnt': test_cnt_list})
            cat_df = pd.concat([base_df.value_counts(), test_df.value_counts()], axis=1).reset_index()
            cat_df.columns = ['bucket', 'base_cnt', 'test_cnt']
            stat_df = pd.concat([stat_df, cat_df])
            stat_df = stat_df.fillna(0)
            stat_df['base_dist'] = stat_df['base_cnt']/len(base_df)
            stat_df['test_dist'] = stat_df['test_cnt']/len(test_df)

        def sub_psi(row):
            base_list = row['base_dist']
            test_list = row['test_dist']

            if base_list==0 and test_list==0:
                return 0
            elif base_list==0 and test_list > 0:
                base_list = 1/len(base_df)
            elif base_list > 0 and test_list==0:
                test_list = 1/len(test_df)
            return (test_list - base_list)* np.log(test_list/base_list)
        
        stat_df['psi'] = stat_df.apply(lambda row: sub_psi(row), axis=1)
        stat_df = stat_df[['bucket', 'base_cnt', 'base_dist', 'test_cnt', 'test_dist', 'psi']]
        psi = stat_df['psi'].sum()
    if out_bins:
        return psi, stat_df, score_bin_list
    else:
        return psi, stat_df





#计算iv
def calculate_iv(x, y, name, max_n_bins=5, special_codes=[], min_prebin_size=0.05, split_digits=6, monotonic_trend ='auto_asc_desc', prebinning_method='cart',dtype='numerical',):
    '''
    x: 特征数据
    y: 标签数据
    name: 特征名称
    max_n_bins: 最大分项数量
    special_codes: 异常值列表
    dtype: 特征类型[numerical, categorical]
    '''
    optb = OptimalBinning(name=name,
                          monotonic_trend=monotonic_trend,
                          dtype=dtype,
                          max_n_bins=max_n_bins,
                          min_prebin_size=min_prebin_size,
                          split_digits=split_digits,
                          special_codes=special_codes,
                          prebinning_method = prebinning_method
                          )
    optb.fit(x, y)
    bin_df = optb.binning_table.build(split_digits)
    IV = bin_df['IV'].max()
    return IV, bin_df, optb


def calculate_iv_splits(x, y, name, max_n_bins=5, special_codes=[], min_prebin_size=0.05,  dtype='numerical'):
    '''
    x: 特征数据
    y: 标签数据
    name: 特征名称
    max_n_bins: 最大分项数量
    special_codes: 异常值列表
    dtype: 特征类型[numerical, categorical]
    '''
    
    min_col = x[~x.isin(special_codes)].min()
    max_col = x[~x.isin(special_codes)].max()
    step_col = (max_col - min_col)/max_n_bins
    user_splits = [(i+1)*step_col for i in range(max_n_bins-1) ]
    user_splits_fixed = [False for i in range(max_n_bins-1)]
    
    optb = OptimalBinning(name=name,
#                           prebinning_method = 'uniform',
                          monotonic_trend=None,
                          dtype=dtype,
                          max_n_bins=max_n_bins,
                          min_prebin_size=min_prebin_size,
                        #   split_digits=split_digits,
                          special_codes=special_codes,
                          user_splits_fixed = user_splits_fixed,
                          user_splits = user_splits
                          )
    optb.fit(x, y)
    bin_df = optb.binning_table.build()
    IV = bin_df['IV'].max()
    return IV, bin_df




#opt分箱
def opt_bins(x_train, x_test, x_oot, y_train, categorical_col, numerical_col, max_n_bins=5, split_digits=4):
    
    x_train_copy = x_train.copy()
    x_test_copy = x_test.copy()
    x_oot_copy = x_oot.copy()

    split_dict = {}
    var_woe_df = pd.DataFrame()

    for variable in categorical_col:
        optb = OptimalBinning(name=variable,
                              monotonic_trend='auto_asc_desc',
                              dtype='categorical',
                              max_n_bins=max_n_bins,
                              min_prebin_size=0.05
                              )
        optb.fit(x_train[variable], y_train)
        x_train_copy[variable] = optb.transform(x_train_copy[variable], metric='woe',
                                                 metric_missing='empirical') * -1
        x_test_copy[variable] = optb.transform(x_test_copy[variable], metric='woe',
                                                metric_missing='empirical') * -1
        x_oot_copy[variable] = optb.transform(x_oot_copy[variable], metric='woe',
                                               metric_missing='empirical') * -1

        split_dict[variable] = [tuple(x) for x in optb.splits]
        bin_df = optb.binning_table.build()
        bin_df['variable'] = variable
        var_woe_df = pd.concat([var_woe_df, bin_df])

    for variable in numerical_col:
        optb = OptimalBinning(name=variable,
                              monotonic_trend='auto_asc_desc',
                              dtype='numerical',
                              max_n_bins=max_n_bins,
                              min_prebin_size=0.05,
                              split_digits=split_digits
                              )
        optb.fit(x_train_copy[variable], y_train)
        x_train_copy[variable] = optb.transform(x_train_copy[variable], metric='woe',
                                                 metric_missing='empirical') * -1
        x_test_copy[variable] = optb.transform(x_test_copy[variable], metric='woe',
                                                metric_missing='empirical') * -1
        x_oot_copy[variable] = optb.transform(x_oot_copy[variable], metric='woe',
                                               metric_missing='empirical') * -1

        split_dict[variable] = list(optb.splits)
        bin_df = optb.binning_table.build(split_digits)
        bin_df['variable'] = variable
        var_woe_df = pd.concat([var_woe_df, bin_df])

    var_woe_df['WoE'] = var_woe_df['WoE'].replace('',np.nan).astype('float')*(-1)
    return x_train_copy, x_test_copy, x_oot_copy, var_woe_df, split_dict




#双向逐步回归
def stepwise_selection(X, y, initial_list=[],threshold_in=0.01,threshold_out = 0.05, verbose = True):
    '''
     threshold_out为t检验，threshold_in为F检验
     X：待筛选变量
     y：好坏标签
    '''
    included = list(initial_list)
 
    while True:
        changed=False
        
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:#F检验最小的显著性小于临界值，说明该变量有显著意义
            # best_feature = new_pval.argmin()
            best_feature = new_pval.index[new_pval.argmin()]
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() 
        if worst_pval > threshold_out:#T检验最大的大于临界值，说明该变量无显著意义。
            changed=True
            # worst_feature = pvalues.argmax()
            worst_feature = pvalues.index[pvalues.argmax()]
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included



#预测结果评估
def binary_metric(target, pred, threshold=0.5):
    '''
    ---二分类数据模型评价函数---
    target: 样本真实标签
    pred: 样本预测概率值
    threshold: 概率阈值
    '''
    pred_ = (pred>=threshold)*1
    accuracy = accuracy_score(target, pred_)     #accuracy_score准确率
    precision = precision_score(target, pred_)   #precision_score精确率
    recall = recall_score(target, pred_)         #recall_score召回率
    f1_measure = f1_score(target, pred_)         #f1_score  f1得分
    confusionMatrix = confusion_matrix(target, pred_)     #confusion_matrix  混淆矩阵
    fpr, tpr, thresholds = roc_curve(target, pred, pos_label=1)   #roc_curve ROC曲线
    auc = roc_auc_score(target, pred)     #roc_auc_score  AUC面积
    KS = max(abs(tpr-fpr))
    TP = confusionMatrix[1, 1]
    FP = confusionMatrix[0, 1]
    FN = confusionMatrix[1, 0]
    TN = confusionMatrix[0, 0]
    lift = (TP/(TP+FP))/((TP+FN)/(TP+FP+FN+TN))
    MAP = average_precision_score(target, pred)    #average_precision_score

#     print ("------------------------- ")
#     print ("confusion matrix:")
#     print ("------------------------- ")
#     print ("| TP: %5d | FP: %5d |" % (confusionMatrix[1, 1], confusionMatrix[0, 1]))
#     print ("----------------------- ")
#     print ("| FN: %5d | TN: %5d |" % (confusionMatrix[1, 0], confusionMatrix[0, 0]))
    print (" ------------------------- ")
#     print ("Accuracy:       %.2f%%" % (accuracy * 100))
#     print ("Recall:         %.2f%%" % (recall * 100))
#     print ("Precision:      %.2f%%" % (precision * 100))
#     print ("F1-measure:     %.2f%%" % (f1_measure * 100))
    print ("AUC:            %.2f%%" % (auc * 100))
    print ("KS:             %.2f%%" % (KS * 100))
#     print ("lift:           %.2f%%" % (lift * 100))
#     print ("MAP:            %.2f%%" % (MAP * 100))
    print ("------------------------- ")
    return ({'AUC':auc, 'KS':KS})



#最终模型分测算，依据qcut自动等频分箱
def qcut_lift_ks(df_cal, score, y, special_list=[], max_bins=20, sort=False, precision=2):
    df = df_cal[(~df_cal[score].isin(special_list)) & (df_cal[score].notnull())]
    if sort==True:#降序排列
        df['bins'] = pd.qcut(df[score],max_bins,duplicates='drop',precision=precision)
        stat = df.groupby(['bins']).agg({'bins':'count', y:'sum'}).sort_index(ascending=False)
    else:
        df['bins'] = pd.qcut(df[score], max_bins, duplicates='drop', precision=precision)
        stat = df.groupby(['bins']).agg({'bins':'count',y:'sum'})
    stat.columns = ['all','bad']
    stat.index = stat.index.astype('str')
    if df_cal[score].isnull().sum()>0:#处理空值
        nan_num = df_cal[score].isnull().sum()
        nan_1_num = df_cal[df_cal[score].isnull()][y].sum()
        nan_df = pd.DataFrame([nan_num,nan_1_num]).T
        nan_df.columns = ['all','bad']
        nan_df.index = ['Nan']
        stat = pd.concat([stat, nan_df])
    if df_cal[score].isin(special_list).sum()>0:
        special_num = df_cal[score].isin(special_list).sum()
        special_1_num = df_cal[df_cal[score].isin(special_list)][y].sum()
        special_df = pd.DataFrame([special_num, special_1_num]).T
        special_df.columns = ['all','bad']
        special_df.index = ['异常']
        stat = pd.concat([stat, special_df])
        
    stat['good'] = stat['all'] - stat['bad']
    stat = stat[['bad', 'good' ,'all']]
    stat['BadRate'] = stat['bad'] / stat['all']
    stat['Bad_CumRate'] = stat['bad'].cumsum() / stat['bad'].sum()
    stat['Good_CumRate'] = stat['good'].cumsum() / stat['good'].sum()
    stat['all_CumRate'] = stat['all'].cumsum() / stat['all'].sum()
    stat['all_Rate'] = stat['all'] / stat['all'].sum()
    
    all_num = stat['all'].sum()
    all_1_num = stat['bad'].sum()
    all_0_num = stat['good'].sum()
    all_bad_rate = all_1_num / all_num
    
    stat['ks'] = abs(stat['Bad_CumRate'] - stat['Good_CumRate'])
    stat['lift'] = stat['BadRate'] / all_bad_rate
    stat['cum_lift'] = (stat['bad'].cumsum()/stat['all'].cumsum()) / all_bad_rate
    
    hj_df = pd.DataFrame([all_1_num, all_0_num, all_num, all_bad_rate, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]).T
    hj_df.columns = ['bad', 'good', 'all', 'BadRate', 'Bad_CumRate', 'Good_CumRate', 'all_CumRate', 
                     'all_Rate', 'ks', 'lift', 'cum_lift']
    hj_df.index=['合计']
    all_stat = pd.concat([stat, hj_df])
    all_stat = all_stat.rename(columns={'bad':'Event', 'good':'NEvent','BadRate':'EventRate', 'Bad_CumRate':'Event_CumRate', 'Good_CumRate':'NEvent_CumRate'})
    return all_stat


#最终模型分测算，依据cut自动等宽分箱
def cut_lift_ks(df_cal, score, y, special_list=[], max_bins=20, sort=False, precision=2):
    df = df_cal[(~df_cal[score].isin(special_list)) & (df_cal[score].notnull())]
    if sort==True:#降序排列
        df['bins'] = pd.cut(df[score],max_bins,duplicates='drop',precision=precision)
        stat = df.groupby(['bins']).agg({'bins':'count', y:'sum'}).sort_index(ascending=False)
    else:
        df['bins'] = pd.cut(df[score], max_bins, duplicates='drop', precision=precision)
        stat = df.groupby(['bins']).agg({'bins':'count',y:'sum'})
    stat.columns = ['all','bad']
    stat.index = stat.index.astype('str')
    if df_cal[score].isnull().sum()>0:#处理空值
        nan_num = df_cal[score].isnull().sum()
        nan_1_num = df_cal[df_cal[score].isnull()][y].sum()
        nan_df = pd.DataFrame([nan_num,nan_1_num]).T
        nan_df.columns = ['all','bad']
        nan_df.index = ['Nan']
        stat = pd.concat([stat, nan_df])
    if df_cal[score].isin(special_list).sum()>0:
        special_num = df_cal[score].isin(special_list).sum()
        special_1_num = df_cal[df_cal[score].isin(special_list)][y].sum()
        special_df = pd.DataFrame([special_num, special_1_num]).T
        special_df.columns = ['all','bad']
        special_df.index = ['异常']
        stat = pd.concat([stat, special_df])
        
    stat['good'] = stat['all'] - stat['bad']
    stat = stat[['bad', 'good' ,'all']]
    stat['BadRate'] = stat['bad'] / stat['all']
    stat['Bad_CumRate'] = stat['bad'].cumsum() / stat['bad'].sum()
    stat['Good_CumRate'] = stat['good'].cumsum() / stat['good'].sum()
    stat['all_CumRate'] = stat['all'].cumsum() / stat['all'].sum()
    stat['all_Rate'] = stat['all'] / stat['all'].sum()
    
    all_num = stat['all'].sum()
    all_1_num = stat['bad'].sum()
    all_0_num = stat['good'].sum()
    all_bad_rate = all_1_num / all_num
    
    stat['ks'] = abs(stat['Bad_CumRate'] - stat['Good_CumRate'])
    stat['lift'] = stat['BadRate'] / all_bad_rate
    stat['cum_lift'] = (stat['bad'].cumsum()/stat['all'].cumsum()) / all_bad_rate
    
    hj_df = pd.DataFrame([all_1_num, all_0_num, all_num, all_bad_rate, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]).T
    hj_df.columns = ['bad', 'good', 'all', 'BadRate', 'Bad_CumRate', 'Good_CumRate', 'all_CumRate', 
                     'all_Rate', 'ks', 'lift', 'cum_lift']
    hj_df.index=['合计']
    all_stat = pd.concat([stat, hj_df])
    all_stat = all_stat.rename(columns={'bad':'Event', 'good':'NEvent','BadRate':'EventRate', 'Bad_CumRate':'Event_CumRate', 'Good_CumRate':'NEvent_CumRate'})
    return all_stat


#最终模型分测算，依据cut自动等宽分箱,可指定分箱节点，也可指定已分箱数据列
def cut_lift_ks1(df_cal, score, y, special_list=[], max_bins=20, sort=False, precision=2, bins_col=''):
    df = df_cal[(~df_cal[score].isin(special_list)) & (df_cal[score].notnull())]
    if bins_col=='':
        df['bins'] = pd.cut(df[score],max_bins,duplicates='drop',precision=precision)
    else:
        df['bins'] = df[bins_col]
    if sort==True:#降序排列
        stat = df.groupby(['bins']).agg({'bins':'count', y:'sum'}).sort_index(ascending=False)
    else:
        stat = df.groupby(['bins']).agg({'bins':'count',y:'sum'})
    stat.columns = ['all','bad']
    stat.index = stat.index.astype('str')
    if df_cal[score].isnull().sum()>0:#处理空值
        nan_num = df_cal[score].isnull().sum()
        nan_1_num = df_cal[df_cal[score].isnull()][y].sum()
        nan_df = pd.DataFrame([nan_num,nan_1_num]).T
        nan_df.columns = ['all','bad']
        nan_df.index = ['Nan']
        stat = pd.concat([stat, nan_df])
    if df_cal[score].isin(special_list).sum()>0:
        special_num = df_cal[score].isin(special_list).sum()
        special_1_num = df_cal[df_cal[score].isin(special_list)][y].sum()
        special_df = pd.DataFrame([special_num, special_1_num]).T
        special_df.columns = ['all','bad']
        special_df.index = ['异常']
        stat = pd.concat([stat, special_df])
        
    stat['good'] = stat['all'] - stat['bad']
    stat = stat[['bad', 'good' ,'all']]
    stat['BadRate'] = stat['bad'] / stat['all']
    stat['Bad_CumRate'] = stat['bad'].cumsum() / stat['bad'].sum()
    stat['Good_CumRate'] = stat['good'].cumsum() / stat['good'].sum()
    stat['all_CumRate'] = stat['all'].cumsum() / stat['all'].sum()
    stat['all_Rate'] = stat['all'] / stat['all'].sum()
    
    all_num = stat['all'].sum()
    all_1_num = stat['bad'].sum()
    all_0_num = stat['good'].sum()
    all_bad_rate = all_1_num / all_num
    
    stat['ks'] = abs(stat['Bad_CumRate'] - stat['Good_CumRate'])
    stat['lift'] = stat['BadRate'] / all_bad_rate
    stat['cum_lift'] = (stat['bad'].cumsum()/stat['all'].cumsum()) / all_bad_rate
    stat['recum_lift'] = (stat.iloc[::-1]['bad'].cumsum()[::-1]/stat.iloc[::-1]['all'].cumsum()[::-1]) / all_bad_rate
    
    hj_df = pd.DataFrame([all_1_num, all_0_num, all_num, all_bad_rate, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]).T
    hj_df.columns = ['bad', 'good', 'all', 'BadRate', 'Bad_CumRate', 'Good_CumRate', 'all_CumRate', 
                     'all_Rate', 'ks', 'lift', 'cum_lift', 'recum_lift']
    hj_df.index=['合计']
    all_stat = pd.concat([stat, hj_df])
    all_stat = all_stat.rename(columns={'bad':'Event', 'good':'NEvent','BadRate':'EventRate', 'Bad_CumRate':'Event_CumRate', 'Good_CumRate':'NEvent_CumRate'})
    return all_stat



#计算头部尾部的lift
def tail_lift(df, y, score, badrate):
    if badrate==0 or df[score].nunique()<=1:
        tail_5p_lift=np.nan
        tail_10p_lift = np.nan
    else:
        tail_5p = df[score].quantile(0.05)
        tail_10p = df[score].quantile(0.1)
        tail_5p_df = df[df[score]<=tail_5p]
        tail_10p_df = df[df[score]<=tail_10p]
        tail_5p_br = tail_5p_df[y].mean()
        tail_10p_br = tail_10p_df[y].mean()
        tail_5p_lift = tail_5p_br/badrate
        tail_10p_lift = tail_10p_br/badrate
    return tail_5p_lift, tail_10p_lift, tail_5p_br, tail_10p_br

def head_lift(df, y, score, badrate):
    if badrate==0 or df[score].nunique()<=1:
        head_5p_lift=np.nan
        head_10p_lift = np.nan
    else:
        head_5p = df[score].quantile(0.95)
        head_10p = df[score].quantile(0.9)
        head_5p_df = df[df[score]>=head_5p]
        head_10p_df = df[df[score]>=head_10p]
        head_5p_br = head_5p_df[y].mean()
        head_10p_br = head_10p_df[y].mean()
        head_5p_lift = head_5p_br/badrate
        head_10p_lift = head_10p_br/badrate
    return head_5p_lift, head_10p_lift, head_5p_br, head_10p_br

def get_tail_head_lift(df, y, score, badrate, sort=False):
    tail_head_lift = pd.Series(index=['尾部5%lift', '尾部10%lift', '头部10%lift', '头部5%lift', '尾部5%EventRate', '尾部10%EventRate', '头部5%EventRate', '头部10%EventRate', ])
    if sort:
        tail_head_lift['尾部5%lift'], tail_head_lift['尾部10%lift'], tail_head_lift['尾部5%EventRate'], tail_head_lift['尾部10%EventRate'] = head_lift(df, y, score, badrate)
        tail_head_lift['头部5%lift'], tail_head_lift['头部10%lift'], tail_head_lift['头部5%EventRate'], tail_head_lift['头部10%EventRate'] = tail_lift(df, y, score, badrate)
    else:
        tail_head_lift['尾部5%lift'], tail_head_lift['尾部10%lift'], tail_head_lift['尾部5%EventRate'], tail_head_lift['尾部10%EventRate'] = tail_lift(df, y, score, badrate)
        tail_head_lift['头部5%lift'], tail_head_lift['头部10%lift'], tail_head_lift['头部5%EventRate'], tail_head_lift['头部10%EventRate'] = head_lift(df, y, score, badrate)
    
    return tail_head_lift

def get_upper_lift(df, y, score, badrate, per_one):
    if badrate==0 or df[score].nunique()<=1:
        upper_br = np.nan
        upper_lift = np.nan
    else:
        cut_score = df[score].quantile(per_one)
        upper_df = df[df[score]>=cut_score]
        upper_br = upper_df[y].mean()
        upper_lift = upper_br/badrate
    return upper_br, upper_lift

def get_lower_lift(df, y, score, badrate, per_one):
    if badrate==0 or df[score].nunique()<=1:
        lower_br = np.nan
        lower_lift = np.nan
    else:
        cut_score = df[score].quantile(per_one)
        lower_df = df[df[score]<=cut_score]
        lower_br = lower_df[y].mean()
        lower_lift = lower_br/badrate
    return lower_br, lower_lift


    

    
def sklearn_show_ks(actual,prob,title='Model'):
    '''
    1.prob为模型的概率;即：prob = rf.predict_proba(X_test)
    2.actual为y_test,真实标签
    '''
    fpr,tpr,threshold = roc_curve(actual,prob)
    ks_ary = list(map(lambda x,y:x-y,tpr,fpr))
    ks = np.max(ks_ary)
    y_axis = list(map(lambda x:x*1.0/len(fpr),range(0,len(fpr))))
    fig = plt.figure(figsize=(8,6))
    plt.title(title + ' ' + "K-S Curve")
    plt.plot(fpr,y_axis,'b--',linewidth=1,label='fpr')
    plt.plot(tpr,y_axis,'y--',linewidth=1,label='tpr')
    plt.plot(y_axis,ks_ary,'g',linewidth=1,label='KS=%.2f'%(ks))
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--',linewidth=1)
    plt.xlim([0,1])
    plt.ylim([0,1])
    return fig


def cumrate_show_ks(df_cal, score, y, max_bins=20, sort=False, title='Model'):
    '''
    使用分箱后好坏累计数量进行作图
    
    '''
    stat = qcut_lift_ks(df_cal, score, y, special_list=[], max_bins=max_bins, sort=sort, precision=2)
    stat = stat.drop(index=['合计'])
    # 'bad':'Event', 'good':'NEvent','BadRate':'EventRate', 'Bad_CumRate':'Event_CumRate', 'Good_CumRate':'NEvent_CumRate'
    stat = stat.rename(columns={'Event_CumRate':'Bad_CumRate', 'NEvent_CumRate':'Good_CumRate'})
    stat = stat[['all_CumRate', 'Bad_CumRate', 'Good_CumRate', 'ks']].reset_index(drop=True)
    df_add = pd.DataFrame({'all_CumRate':[0], 'Bad_CumRate':[0], 'Good_CumRate':[0], 'ks':[0]})
    stat = pd.concat([df_add, stat]).reset_index(drop=True)
    
    fpr,tpr,threshold = roc_curve(df_cal[y],df_cal[score])
    ks = max(abs(tpr-fpr))
    x_new = np.linspace(stat['all_CumRate'].min(), stat['all_CumRate'].max(), 300)
    bad_smooth = make_interp_spline(stat['all_CumRate'], stat['Bad_CumRate'])(x_new)
    good_smooth = make_interp_spline(stat['all_CumRate'], stat['Good_CumRate'])(x_new)
    ks_smooth = make_interp_spline(stat['all_CumRate'], stat['ks'])(x_new)
    
    fig = plt.figure(figsize=(8,6))
    plt.title(title + ' ' + "K-S Curve")
    plt.plot(x_new, bad_smooth,'b-',linewidth=1,label='Event_CumRate')
    plt.plot(x_new, good_smooth,'y-',linewidth=1,label='NonEvent_CumRate')
    plt.plot(x_new, ks_smooth,'g',linewidth=1,label='KS=%.2f'%(ks))
    plt.legend(loc='lower right')
#     plt.plot([0,1],[0,1],'r--',linewidth=1)
    plt.xlabel('CumRate_All', fontsize=14)
    plt.ylabel('CumRate(Event/NonEvent)', fontsize=14)
    plt.xlim([0,1])
    plt.ylim([0,1])
    return fig



    
#画分布图
def score_distribution(score_df, score, y, bins=10, xlim='', fig_title='Score Distribution'):
    score_df_cal = score_df.copy()
    score_df_cal['cut'] = pd.cut(score_df_cal[score], bins=bins)
    stat = score_df_cal.groupby('cut').agg({score:'median', y:'mean'})
    
    fig, ax1 = plt.subplots(figsize=(8,6))
#     color = 'tab:green'
    ax1.set_title(fig_title, fontsize=16)
    ax1.set_xlabel('Score', fontsize=14)
    ax1.set_ylabel('Count', fontsize=14)
    if xlim != '':
        ax1.set_xlim(eval(xlim))
    #第一图条形图
    ax1 = sns.histplot(x=score, data=score_df, bins=10, palette='summer')
    ax1.tick_params(axis='y')
    #twinx共享x轴(类似的语法，如共享y轴twiny)
    ax2 = ax1.twinx()
    color = 'tab:red'
    #第二个图，折线图
    ax2.set_ylabel('EventRate', fontsize=14)
    ax2 = sns.lineplot(x=score, y=y, data=stat, color=color)
    ax2.tick_params(axis='y', color=color)
    
    return fig    
    

    
    
def model_score_report(score_df, score, y, auc_score='', max_bins=10, precision=2, sort=False, special_list=[], self_bins=[], upperlift_per=[], lowerlift_per=[]):
    '''
    score_df: DataFrame格式
    score: 模型分列名,score的非空不同取值>1,且为数值型
    y: y标名称，y标不能为空，且为0/1二分类
    sort: 模型分是否为越大越好，默认为True
    special_list: 特殊取值列表
    upperlift_per: 大于某分位点的百分比, 浮点型
    lowerlift_per: 小于某分位点的百分比, 浮点型
    '''
    score_infos = {}
    
    score_df_n = score_df[(score_df[score].notnull()) & (~score_df[score].isin(special_list))]
    if auc_score=='':
        score_infos['AUC'] = roc_auc_score(score_df_n[y], score_df_n[score])
    else:
        score_infos['AUC'] = roc_auc_score(score_df_n[y], score_df_n[auc_score])
    fpr, tpr, thresholds = roc_curve(score_df_n[y], score_df_n[score], pos_label=1)   #roc_curve ROC曲线
    score_infos['KS'] = max(abs(tpr-fpr))
    
    #头尾lift
    badrate = score_df[y].mean()
    score_infos['TH_lift'] = get_tail_head_lift(score_df_n, y, score, badrate, sort=sort)
    
    #等频分箱
    score_infos['qcut_ks_lift'] = qcut_lift_ks(score_df, score, y, special_list=special_list, 
                                          max_bins=max_bins, sort=sort, precision=2)
    #等宽分箱
    if len(self_bins)==0:
        score_infos['cut_ks_lift'] = cut_lift_ks(score_df, score, y, special_list=special_list, 
                                                 max_bins=max_bins, sort=sort, precision=precision)
    else:
        score_infos['cut_ks_lift'] = cut_lift_ks(score_df, score, y, special_list=special_list, 
                                                 max_bins=self_bins, sort=sort, precision=precision)
        
    if len(upperlift_per)>0:
        upperlift = {}
        for per_one in upperlift_per:
            _, upperlift['大于'+str(per_one)+'lift'] = get_upper_lift(score_df_n, y, score, badrate, per_one)
        score_infos['upper_lift'] = upperlift
    if len(lowerlift_per)>0:
        lowerlift = {}
        for per_one in lowerlift_per:
            _, lowerlift['小于'+str(per_one)+'lift'] = get_lower_lift(score_df_n, y, score, badrate, per_one)
        score_infos['lower_lift'] = lowerlift
    
    return score_infos




#概率评分转换
def scoretofico(pred, Pdo, Base, Odds):
    '''
    phat_final: 预测概率
    '''
    logit_final = np.log(pred / (1 - pred))
    B = (Pdo) / np.log(2)  # Factor - B
    A = Base + (B * np.log(Odds))  # Offset - A    Base - 基础分

    APScore = round(A - B * logit_final,0)

    return APScore



#optuna调参过程结果保存
from datetime import datetime
class TrialSaver:
    def __init__(self, output_dir="./results", experiment_name="exp"):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.filename = f"{output_dir}/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    def __call__(self, study, trial):
    
        # 构造一行数据：包含 trial number、所有超参、目标值
        # row = {"trial_number": trial.number, 'trial_params':trial.params}
        row = {"trial_number": trial.number, 'trial_params':trial.params}
        row.update(trial.user_attrs)
        # print(trial.params)
        
        # 转为 DataFrame
        df = pd.DataFrame([row])
        
        #追加模式
        if not os.path.exists(self.filename):
            df.to_csv(self.filename, index=False)
        else:
            df.to_csv(self.filename, mode='a', header=False, index=False)




#训练样本信息统计脚本
def calc_binary_stats_multi(
    dfs: Union[List[pd.DataFrame], Dict[str, pd.DataFrame]],
    y_label: str,
    pos_label=1,
    na_values=None,  # 可选：额外指定哪些值视为空（如 ["", "N/A"]）
    stat_missing = False
    ) -> pd.DataFrame:
    """
    对多个 DataFrame 计算二分类标签的样本统计信息，自动忽略标签为空的样本。
    
    空值包括：np.nan, None, pd.NA, pd.NaT，以及可选的自定义空值。
    
    Parameters:
    ----------
    dfs : list 或 dict of DataFrames
    y_label : str
        标签列名
    pos_label : int/str, default=1
        正样本的取值
    na_values : list, optional
        额外视为空值的值，如 ["", "NULL"]
        
    Returns:
    -------
    pd.DataFrame : 包含 total（有效样本数）、positive_count、positive_rate
    """
    results = {}

    if isinstance(dfs, dict):
        named_dfs = dfs
    elif isinstance(dfs, list):
        named_dfs = {f"df_{i}": df for i, df in enumerate(dfs)}
    else:
        raise TypeError("`dfs` 必须是 list 或 dict")

    for name, df in named_dfs.items():
        if y_label not in df.columns:
            raise ValueError(f"DataFrame '{name}' 缺少列 '{y_label}'")
        
        y = df[y_label].copy()

        # Step 1: 标记空值（内置 NaN/None + 自定义空值）
        is_na = y.isna()  # 处理 np.nan, None, pd.NA 等
        
        if na_values:
            # 将用户指定的值也标记为空
            for val in na_values:
                is_na = is_na | (y == val)
        
        # Step 2: 只保留非空样本
        y_valid = y[~is_na]
        
        total_valid = len(y_valid)
        if total_valid == 0:
            pos_count = 0
            pos_rate = 0.0
        else:
            pos_count = (y_valid == pos_label).sum()
            pos_rate = pos_count / total_valid

        if stat_missing:
            results[name] = {
                "total_valid": int(total_valid),      # 有效样本数（非空标签）
                "pos_count": int(pos_count),
                "pos_rate": float(pos_rate),
                "missing_count": int(is_na.sum())     # 可选：记录缺失数
            }
        else:
            results[name] = {
                "total_valid": int(total_valid),      # 有效样本数（非空标签）
                "pos_count": int(pos_count),
                "pos_rate": float(pos_rate)
            }

    result_df = pd.DataFrame(results).T
    # result_df.index.name = "dataset"
    return result_df
