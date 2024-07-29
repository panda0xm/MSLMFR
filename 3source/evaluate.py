from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import pandas as pd 
import numpy as np
def kmeans_acc(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    row_ind,col_ind = linear_sum_assignment(-cm)
    acc = cm[row_ind,col_ind].sum() / len(y_true)    
    return acc

def kmeans_purity(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return metrics.accuracy_score(y_true, y_voted_labels)

def evaluate(y_true, y_pred):
    acc=kmeans_acc(y_true, y_pred)
    nmi=metrics.normalized_mutual_info_score(y_true, y_pred)
    purity=kmeans_purity(y_true, y_pred)
    f1_score=0
    precision=0
    recall=0
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    
    # 假设聚类结果为predicted_labels，真实类别标签为true_labels

    # 创建一个矩阵，其中元素m[i, j]表示聚类结果中标签i与真实标签j的交集大小
    n_clusters = len(set(y_pred))
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred, sparse=False)
    #print(contingency_matrix)
    #cm = metrics.confusion_matrix(y_true, y_pred)
    #print(cm)
    # 使用匈牙利算法将矩阵中的元素最大化，找到最优的聚类-真实类别匹配
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    # 对聚类结果进行重新标记，使其与真实类别标签一一对应
    y_pred2 = np.zeros_like(y_pred)
    for j, i in zip(row_ind, col_ind):
        y_pred2[y_pred == i] = j   
    #print(row_ind)
    #print(col_ind)
    #print(y_pred)
    pd.set_option('display.max_columns', None)    # 显示所有列
    pd.set_option('display.max_rows', None)      # 显示所有行
    np.set_printoptions(threshold=np.inf)
    #print(y_pred2)
    #cm = metrics.confusion_matrix(y_true, y_pred2)
    #print(cm)
    # 根据最优匹配计算F1分数
    f1_score = metrics.f1_score(y_true, y_pred2,average='macro')
    #print("F1 Score:", f1_score)

    # 根据最优匹配计算精确度
    precision = metrics.precision_score(y_true, y_pred2,average='macro')
    #print("Precision:", precision)

    # 根据最优匹配计算召回率
    recall = metrics.recall_score(y_true, y_pred2,average='macro')
    #print("Recall:", recall)
    return acc,nmi,purity,f1_score,precision,recall,ari