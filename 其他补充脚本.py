import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


def build_complete_groups_with_fixed(
    all_features: List[str],
    user_groups: Dict[str, List[str]],
    fixed_features: List[str],
    singleton_prefix: str = "singleton"
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    构建完整分组，并分离出固定特征。
    
    Returns:
    -------
    ablation_groups : 用于剔除实验的分组（不含 fixed_features）
    final_all_features : 所有可变 + 固定特征（用于检查）
    """
    all_features = list(all_features)
    fixed_features = list(fixed_features) if fixed_features else []
    
    # 检查 fixed_features 是否在 all_features 中
    for f in fixed_features:
        if f not in all_features:
            raise ValueError(f"fixed_features 中的 '{f}' 不在 all_features 中")
    
    # 可被剔除的特征 = 全部 - 固定
    ablatable_features = [f for f in all_features if f not in fixed_features]
    
    # 收集用户已分组的可剔除特征
    grouped_in_ablatable = set()
    clean_user_groups = {}
    for group_name, feats in user_groups.items():
        valid_feats = []
        for f in feats:
            if f in fixed_features:
                raise ValueError(f"用户组 '{group_name}' 包含 fixed_features 中的特征: {f}")
            if f not in ablatable_features:
                raise ValueError(f"用户组 '{group_name}' 包含未知特征: {f}")
            valid_feats.append(f)
            grouped_in_ablatable.add(f)
        if valid_feats:
            clean_user_groups[group_name] = valid_feats
    
    # 未分组的可剔除特征 → 单例组
    ungrouped = [f for f in ablatable_features if f not in grouped_in_ablatable]
    singleton_groups = {f"{singleton_prefix}_{f}": [f] for f in ungrouped}
    
    ablation_groups = {**clean_user_groups, **singleton_groups}
    
    # 验证覆盖
    covered = []
    for feats in ablation_groups.values():
        covered.extend(feats)
    assert len(covered) == len(set(covered)) == len(ablatable_features), "可剔除特征覆盖不完整！"
    
    return ablation_groups, all_features


def evaluate_group_ablation_with_fixed(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    oot_dfs: List[pd.DataFrame],
    all_features: List[str],
    target_col: str,
    lgb_params: Dict,
    best_index : str,
    eval_index : List[str],
    categorical_features: Optional[List[str]] = None,
    user_groups: Optional[Dict[str, List[str]]] = None,
    fixed_features: Optional[List[str]] = None,
    oot_names: Optional[List[str]] = None,
    
    verbose: bool = True
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    按组剔除特征实验，支持指定永不剔除的特征。
    
    Parameters:
    ----------
    all_features : 所有候选建模特征（包括 fixed 和可剔除）
    user_groups : 用户定义的特征组（仅限可剔除特征）
    fixed_features : 永不剔除的特征列表（始终保留）
    train_df, val_df, oot_dfs : 数据集
    target_col : 标签列名
    lgb_params : LightGBM 参数
    oot_names : OOT 数据集名称（可选）
    top_pct : Lift 计算的头部比例
    verbose : 是否打印进度
    categorical_features : 预处理好的类别型变量
    eval_index : 需计算的评估指标，不可为空，默认AUC，当前可选['auc', 'ks']
    best_index : 选择最优评估指标，由数据集标识和评估指标两部分构成，当前可选指标auc、ks，默认数据集为验证集val，该参数默认val_auc
    """
    if categorical_features is None:
        categorical_features = []
    if user_groups is None:
        user_groups = {}
    if fixed_features is None:
        fixed_features = []
    if oot_names is None:
        oot_names = [f"oot_{i}" for i in range(len(oot_dfs))]
    elif len(oot_names) != len(oot_dfs):
        raise ValueError("oot_names 长度需与 oot_dfs 一致")
    
    # 构建剔除分组（排除 fixed_features）
    ablation_groups, _ = build_complete_groups_with_fixed(
        all_features, user_groups, fixed_features
    )
    
    all_datasets = {
        "train": train_df,
        "val": val_df,
        **dict(zip(oot_names, oot_dfs))
    }

    results = []

    # === 基线：全特征（fixed + 所有可剔除）===
    baseline_features = list(all_features)  # fixed + ablatable
    cat_feats_in_train = [f for f in categorical_features if f in baseline_features]
    if verbose:
        print("Running baseline (all features)...")
    metrics_all = _evaluate_features_with_fixed(
        train_df, val_df, baseline_features, target_col, lgb_params, all_datasets, cat_feats_in_train, eval_index
    )
    metrics_all["removed_group"] = "NONE"
    results.append(metrics_all)

    # === 依次剔除每个组（但 fixed_features 始终保留）===
    for group_name, group_features in ablation_groups.items():
        if verbose:
            print(f"Removing group: {group_name} → features {group_features}")
        
        # 特征集 = fixed_features + (ablatable_features - group_features)
        remaining_ablatable = [f for f in all_features if f not in group_features and f not in fixed_features]
        current_features = fixed_features + remaining_ablatable
        cat_feats_in_train = [f for f in categorical_features if f in current_features]
        
        if len(current_features) == 0:
            if verbose:
                print("  ⚠️ 无特征剩余，跳过")
            continue
        
        metrics = _evaluate_features_with_fixed(
            train_df, val_df, current_features, target_col, lgb_params, all_datasets, cat_feats_in_train, eval_index
        )
        metrics["removed_group"] = group_name
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.set_index("removed_group", inplace=True)

    best_group = results_df[best_index].idxmax()
    best_removed = best_group if best_group != "NONE" else None

    return results_df, best_removed


def _evaluate_features_with_fixed(train_df, val_df,  features, target_col, lgb_params, all_datasets, cat_feats_in_train, eval_index):
    """内部评估函数（同前，略作简化）"""
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_val = val_df[features]
    y_val = val_df[target_col]

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feats_in_train)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_feats_in_train, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        verbose_eval=False,
        valid_sets=[val_data],
        early_stopping_rounds=20
    )

    metrics = {}
    for name, df in all_datasets.items():
        if len(df) == 0:
            continue
        X = df[features]
        y = df[target_col]
        valid_mask = y.notna()
        y_clean = y[valid_mask]
        pred = model.predict(X.loc[valid_mask])
        if 'auc' in eval_index:
            try:
                auc_tmp = roc_auc_score(y_clean, pred) 
            except:
                auc_tmp = np.nan
            metrics[f"{name}_auc"] = auc_tmp

        if 'ks' in eval_index:
            try:
                fpr_tmp,tpr_tmp,_ = roc_curve(y_clean, pred)
                ks_tmp = max(abs(fpr_tmp-tpr_tmp))
            except:
                ks_tmp = np.nan
            metrics[f"{name}_ks"] = ks_tmp

    return metrics


saver = TrialSaver(output_dir="./", experiment_name="lgbm_v1")

fea_obj = feas_round1.copy()
fea_cat_obj = []
study = optuna.create_study(directions=["maximize", 'minimize'])
# study = optuna.create_study(directions=["maximize"])
func = lambda trial: objective(trial, train[fea_obj], y_train, test[fea_obj], y_test, df_oot[fea_obj], df_oot[y_label], df_oot1[fea_obj], df_oot1[y_label], fea_cat_obj)
# study.optimize(func, n_trials=5, callbacks=[save_trial_callback])
study.optimize(func, n_trials=5, callbacks=[saver])






# lgbm解析脚本调用案例
parser = LGBMTreeParser(lgbm)
df_nodes = parser.get_nodes_with_sample_counts(test[feas])
f_map = {i: name for i, name in enumerate(lgbm.feature_name())}
df_nodes['feature_name'] = df_nodes['split_feature'].map(f_map)
print("\n=== 基于 'mean_concave_points' 的分裂情况 ===")
target_feat = 'mean_concave_points'
splits = df_nodes[
    (~df_nodes['is_leaf']) & 
    (df_nodes['feature_name'] == target_feat)
]
if not splits.empty:
    print(splits[['tree_index', 'depth', 'threshold', 'gain']].head())

# 2.3 可视化第一棵树
print("\n=== 第0棵树的结构可视化 ===")
parser.print_tree_structure(tree_index=0)








