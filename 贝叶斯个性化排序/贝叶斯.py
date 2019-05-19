import tensorflow as tf
import numpy
import os
import random
from collections import defaultdict

#tensorflow实现BPR
def load_data(data_path):
    user_ratings = defaultdict(set)#set：集合，集合无重复
    max_u_id = -1
    max_i_id = -1
    with open(data_path,'r') as f:
        for lines in f.readlines():
            u,i,_,_ = line.split("\t")
            u = int(u)
            i = int(i)
            user_ratings[u].add(i)
            max_u_id = max(u,max_u_id)
            max_i_id = max(i,max_i_id)

    print("max_u_id:",max_u_id)
    print("max_i_id:",max_i_id)
    return max_u_id,max_i_id,user_ratings

data_path = os.path.join('D:\\tmp\\ml-100k', 'u.data')
user_count, item_count, user_ratings = load_data(data_path)#输出用户数和电影数，同时把每个用户看过的电影都保存在user_ratings中
#数据集 max_u_id = 943,max_i_id = 1682

#对每一个用户u，在user_rating中随机找到他评分过的一部电影i，保存在user_ratings_test中
def generate_test(user_ratings):
    user_test = dict()#生成一个字典
    for u,i_list in user_ratings.items():#?
        user_test[u] = random.sample(user_ratings,1)[0]#[0]是为了把元素提取出来
    return user_test
user_ratings_test = generate_test(user_ratings)#得到一个评分过的电影

#用tensorflow迭代用的若干批训练集,根据user_ratings找到若干训练用的三元组<u,i,j>
#用户u随机抽取，i从user——ratings中随机抽取，j从总的电影集中抽取，但（u,j）不在user_ratings中
#构造训练集三元组<u,i,j>
def generate_train_batch(user_ratings,user_rating_test,item_count,batch_size = 512):
    t = []
    for b in range(batch_size):
        u = random.sample(user_ratings.keys(),1)[0]
        i = random.sample(user_ratings[u],1)[0]

        while i == user_ratings_test[u]:
            i = random.sample(user_ratings[u],1)[0]

        j = random.randint(1,item_count)
        while j in user_ratings[u]:
            j = random.randint(1,item_count)#返回item_count个0-1的数
        t.append([u,i,j])
    return numpy.asarray(t)

#测试集三元组<u,i,j>
#i从user_ratings_test中随机抽取，j是u没有评分过的电影
def generate_test_batch(user_ratings,user_ratings_test,item_count，batch_size = 512):
    for u in user_ratings.keys():
        t = []
        i = user_rating_test[u]
        for j in range(1,item_count + 1):
            if not(j in user_ratings[u]):
                t.append([u,i,j])
        yield numpy.asarray(t)

#tensorflow实现
def bpr_mf(user_count,item_count,hidden_dim):#hidden_dim为隐含维度k
    u = tf.placeholder(tf.int32,[None])
    i = tf.placeholder(tf.int32, [None])
    j = tf.placeholder(tf.int32, [None])

    with tf.device("/cpu:0"):#选择CPU
        #建立变量op
        user_emb_w = tf.get_variable("user_emb_w",[user_count + 1,hidden_dim],initializer = tf.random_normal_initializer(0,0.1))
        item_emb_w = tf.get_variable("item_emb_w",[item_count + 1,hidden_dim],initializer = tf.random_normal_initializer(0,0.1))

        u_emb = tf.nn.embedding_lookup(user_emb_w,u)
        i_emb = tf.nn.embedding_lookup(item_emb_w,i)
        j_emb = tf.nn.embedding_lookup(item_emb_w,j)

    #MF predict : u_i > u_j
    #multiply为点乘
    x = tf.reduce_sum(tf.multiply(u_emb,(i_emb - j_emb)),1,keep_dims=True)


