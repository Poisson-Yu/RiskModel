import imp
import pandas as pd
import statsmodels.api as sm

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
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
 
        model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
       
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() 
        if worst_pval > threshold_out:#T检验最大的大于临界值，说明该变量无显著意义。
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included