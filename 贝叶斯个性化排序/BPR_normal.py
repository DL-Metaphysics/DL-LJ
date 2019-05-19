#_autor: Thinkpad
#date: 2019/5/19

import random
import scores
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score

def pre_handel(set,predict,item_count):
    for u in set.keys():
        for j in set[u]:
            predict[(u-1) * item_count + j - 1] = 0
    return predict

def load_data(path):
    user_ratings = defaultdict(set)#得到用户集
    max_u_id = -1
    max_i_id = -1
    with open(path, 'r') as f:
        for line in f.readlines():
            u, i = line.split(" ")
            u = int(u)
            i = int(i)
            user_ratings[u].add(i)
            max_u_id = max(u, max_u_id)
            max_i_id = max(i, max_i_id)
    return user_ratings,max_u_id,max_i_id

data_path = os.path.join('D:\\tmp\\ml-100k', 'u.data')
user_count, item_count, user_ratings = load_data(data_path)

def load_test_data(path,user_count,item_count):
    file = open(path,'r')
    test_data = np.zeros((user_count, item_count))
    for line in file:
        line = line.split('')
        user = int(line[0])
        item = int(line[1])
        test_data[user - 1][item - 1] = 1
    return test_data

#根据user_ratings找到若干训练用的三元组<u,i,j>
#用户u随机抽取，i从user——ratings中随机抽取，j从总的电影集中抽取，但（u,j）不在user_ratings中
#构造训练集三元组<u,i,j>
def train(user_ratings_train,user_count,item_count,alpha = 0.01,lam = 0.01):
    U = np.random.rand(user_count, latent_factors) * 0.01
    V = np.random.rand(item_count, latent_factors) * 0.01
    biasV = np.random.rand(item_count) * 0.01
    latent_factors = 20
    train_count = 1000

    for user in range(user_count):
        u = random,randint(1,user_count)#返回闭区间[1，user_count]的一个随机值
        if u not in user_ratings_train.keys():
            continue
        i = random.sample(user_ratings_train[u],1)[0]#随机抽取一个u用户评分过的电影i
        j = random.randint(1,item_count)
        while j in user_ratings_train[u]:#保证i和j不可以相同
            j = random.randint(1,item_count)

        u -= 1
        i -= 1
        j -= 1
        r_ui = np.dot(U[u],V[i].T) + biasV[i]
        r_uj = np.dot(U[u],V[j].T) + biasV[j]
        r_uij = r_ui - r_uj
        loss_func = -1.0 / (1 + np.exp(r_uij))

        #更新 U和V
        U[u] += -alpha * (loss_func * (V[i] - V[j]) + lam * U[u])
        V[i] += -alpha * (loss_func * U[u] + lam * V[i])
        V[j] += -alpha * (loss_func * (-U[u]) + lam * V[j])
        #更新 biasV
        self.biasV[i] += -self.lr * (loss_func + self.reg * self.biasV[i])
        self.biasV[j] += -self.lr * (-loss_func + self.reg * self.biasV[j])

#
def predict(user,item):
    predict = np.mat(user) * np.mat(item.T)
    return predict

if _name_ == '_main_':
    user_ratings_train, user_count,item_count = load_data('BPR_MPR/train.txt')
    test_data = load_test_data(path,user_count,item_count)

