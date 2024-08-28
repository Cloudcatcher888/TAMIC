import os
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.nn.rnn_cell import GRUCell


class Model(object):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="DNN"):
        self.model_flag = flag
        self.reg = False
        self.batch_size = batch_size
        self.n_mid = n_mid
        self.n_cate = 1600
        self.neg_num = 10
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.t_batch_ph = tf.placeholder(tf.float32, [None, ], name='t_batch_ph')
            self.t_his_batch_ph =tf.placeholder(tf.float32, [None, None], name='t_his_batch_ph')
            # self.cate_batch_ph = tf.placeholder(tf.int32, [None, ], name='t_batch_ph')
            # self.cate_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='t_his_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])

        self.mask_length = tf.cast(tf.reduce_sum(self.mask, -1), dtype=tf.int32)

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, embedding_dim], trainable=True)
            self.mid_embeddings_bias = tf.get_variable("bias_lookup_table", [n_mid], initializer=tf.zeros_initializer(), trainable=False)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)



        self.item_eb = self.mid_batch_embedded
        self.item_his_eb = self.mid_his_batch_embedded * tf.reshape(self.mask, (-1, seq_len, 1))


    def build_sampled_softmax_loss(self, item_emb, user_emb, loss_reg=None):
        if loss_reg is not None:
            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias, tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb, self.neg_num * self.batch_size, self.n_mid)) + loss_reg
        else:
            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias, tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb, self.neg_num * self.batch_size, self.n_mid))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.mid_his_batch_ph: inps[2],
            self.t_batch_ph: inps[3],
            self.t_his_batch_ph: inps[4],
            # self.cate_batch_ph: inps[5],
            # self.cate_his_batch_ph: inps[6],
            self.mask: inps[5],
            self.lr: inps[6]
        }
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def output_item(self, sess):
        item_embs = sess.run(self.mid_embeddings_var)
        return item_embs

    def output_user(self, sess, inps):
        user_embs = sess.run(self.user_eb, feed_dict={
            self.mid_his_batch_ph: inps[0],
            self.t_batch_ph: inps[1],
            self.t_his_batch_ph: inps[2],
            # self.cate_his_batch_ph: inps[3],
            self.mask: inps[3]
        })
        return user_embs
    
    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)

class Model_DNN(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(Model_DNN, self).__init__(n_mid, embedding_dim, hidden_size,
                                           batch_size, seq_len, flag="DNN")

        masks = tf.concat([tf.expand_dims(self.mask, -1) for _ in range(hidden_size)], axis=-1)

        self.item_his_eb_mean = tf.reduce_sum(self.item_his_eb, 1) / (tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9)
        self.user_eb = tf.layers.dense(self.item_his_eb_mean, hidden_size, activation=None)
        self.build_sampled_softmax_loss(self.item_eb, self.user_eb)

class Model_GRU4REC(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(Model_GRU4REC, self).__init__(n_mid, embedding_dim, hidden_size,
                                           batch_size, seq_len, flag="GRU4REC")
        with tf.name_scope('rnn_1'):
            self.sequence_length = self.mask_length
            rnn_outputs, final_state1 = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.item_his_eb,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="gru1")

        self.user_eb = final_state1
        self.build_sampled_softmax_loss(self.item_eb, self.user_eb)


def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape

class TimeEncoder():
    def __init__(self,maxtime=1406073600,mintime=832550400):
        self.maxtime = maxtime
        self.mintime = mintime
        self.span = 10000
        self.max_time_span = int(self.maxtime/self.span) + 3
        self.time_emb_dim = 100
        time_encode = []
        position = list(range(0, self.max_time_span))
        div_term = [math.exp(i * -(math.log(10000.0) / self.time_emb_dim)) for i in range(int(self.time_emb_dim/2))]
        for p in position:
            te = []
            for d in div_term:
                te.append(math.sin(p*d))
                te.append(math.cos(p*d))
            time_encode.append(te)
        self.time_encode = tf.constant(time_encode, dtype=tf.float32)

    def get_time_encode(self, timestamps):
        idx = tf.cast(timestamps/self.span, dtype=tf.int32)
        return tf.nn.embedding_lookup(self.time_encode, idx-tf.reduce_min(idx))


class CapsuleNetwork(tf.layers.Layer):
    def __init__(self, m_dim, dim, seq_len, bilinear_type=2, num_interest=4, hard_readout=True, relu_layer=False,tmax=100,tmin=100):
        super(CapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.num_interest = num_interest
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True
        self.time_encoder = TimeEncoder(tmax,tmin)
        print(tmax,tmin)

        self.n_gate = 8
        self.alpha = 0.1
        trd_phase = [0,100000]
        init_phase = lambda shape, dtype, partition_info=None: trd_phase[1]*np.random.rand(*shape)
        self.phase = tf.get_variable("phase", [1,1, self.n_gate], trainable=True,constraint=lambda x: tf.clip_by_value(x,*trd_phase),initializer=init_phase)
        # trd_tao = [10000, 10000000]
        # init_tao = lambda shape, dtype, partition_info=None: trd_tao[1] * np.random.rand(*shape)
        # self.tao = tf.get_variable("tao", [1,1, self.n_gate], trainable=True,constraint=lambda x: tf.clip_by_value(x, *trd_tao),initializer=init_tao)

        self.tao = tf.get_variable("tao", [1, 1, self.n_gate], trainable=False,initializer=tf.constant_initializer(86400*np.array([[[1,2,7,14,30,60,365,730]]])))
        # self.tao = tf.get_variable("tao", [1, 1, self.n_gate], trainable=False,initializer=tf.constant_initializer(86400*np.array([[[1,2,7,14,30,60,365,730]]])))
        # self.tao = tf.get_variable("tao", [1, 1, self.n_gate], trainable=False,initializer=tf.constant_initializer(86400*np.array([[[2,7,14,30,60,365]]])))
        # self.tao = tf.get_variable("tao", [1, 1, self.n_gate], trainable=False,initializer=tf.constant_initializer(86400*np.array([[[7,14,30,365]]])))

        trd_ratio = [0, 0.3]
        init_ratio = lambda shape, dtype, partition_info=None: trd_ratio[1] * np.random.rand(*shape)
        self.ratio = tf.get_variable("ratio", [1,1,self.n_gate], trainable=True,constraint=lambda x: tf.clip_by_value(x, *trd_ratio), initializer=init_ratio)
        # trd_phase = [0, 100000]
        # init_phase = lambda shape, dtype: trd_phase * np.random.rand(*shape)
        # self.lamda = tf.get_variable("lamda", [1,1,1], trainable=True)
        self.gate_embeddings_var = tf.get_variable("gate_embedding_var", [1, m_dim, self.n_gate], trainable=True)

    def call(self, item_his_emb, item_eb, t_batch_ph, t_his_batch_ph, mask, ta):

        t_batch_ph = tf.tile(tf.expand_dims(t_batch_ph,axis=1),[1,self.seq_len])
        t_emb_batch = self.time_encoder.get_time_encode(t_batch_ph)
        t_emb_his_batch = self.time_encoder.get_time_encode(t_his_batch_ph)
        delta_t_emb_batch = self.time_encoder.get_time_encode((t_batch_ph - t_his_batch_ph))
        time_feature = tf.concat([t_emb_batch, t_emb_his_batch, delta_t_emb_batch], axis=2)
        # print_op = tf.print(time_feature)
        hidden1 = tf.layers.dense(time_feature,100,activation=tf.nn.relu, name='t_layer1')
        t_factor = tf.layers.dense(hidden1,1,activation=tf.nn.sigmoid, name='t_layer2')
        t_factor = tf.transpose(t_factor,[0,2,1])

        #g_factor
        phi = tf.mod((tf.expand_dims(t_his_batch_ph,axis=2) - self.phase),self.tao)/self.tao
        # gate = 2*phi/self.ratio*tf.sign(tf.maximum(phi - 0.5*self.ratio, 0))\
        #        +(2-2*phi/self.ratio)*tf.sign(tf.maximum(phi>0.5*self.ratio and phi<self.ratio, 0))\
        #        +self.alpha*phi*tf.sign(tf.maximum(phi-self.ratio, 0))

        gate = tf.where(tf.less(phi,self.ratio),tf.maximum(tf.sin(3.14159*phi/self.ratio),0), self.alpha*phi)
        gate_ratio = tf.multiply(tf.nn.softmax(tf.matmul(item_his_emb,self.gate_embeddings_var),axis=2), gate)
        g_factor = tf.expand_dims(tf.reduce_sum(gate_ratio, axis=2),axis=1)
        # with tf.variable_scope('t_layer1',reuse=True):
        #     print_op = tf.print(tf.get_variable('kernel'))
        reg1 = tf.norm(t_factor)
        reg2 = tf.norm(self.ratio)+tf.norm(self.phase)/100000
        loss_reg = 0.1*reg1 + 1*reg2 + 1/reg1 + 0.1/reg2
        print_ops = []
        # print_ops.append(tf.print(t_his_batch_ph, 't_his'))
        # print_ops.append(tf.print(g_factor,'g_factor'))
        # print_ops.append(tf.print(t_factor,'t_factor'))
        # print_ops.append(tf.print(phi, 'phi'))
        # print_ops.append(tf.print(self.ratio, 'ratio'))
        # print_ops.append(tf.print(gate, 'gate'))
        # print_ops.append(tf.print(self.tao, 'tao'))
        # print_ops.append(tf.print(self.phase, 'phase'))
        print_ops.append(tf.print(loss_reg,'loss_reg'))
        # print_ops.append(tf.print(tf.norm(t_factor),'t_factor_norm'))
        # print_ops.append(tf.print(tf.norm(self.ratio),'ratio_norm'))
        # print_ops.append(tf.print(tf.norm(self.phase)/100000,'phase_norm'))
        # print_ops.append(tf.print(tf.norm(self.tao)/100000000,'tao_norm'))
        if ta==1:
            loss_reg = 1 * reg1 + 1 / reg1
            factor = tf.tile(t_factor, [1, self.num_interest, 1])
        if ta==2:
            loss_reg = 0.1 * reg1 + 1 * reg2 + 1 / reg1 + 0.1 / reg2
            factor = tf.tile(t_factor+g_factor,[1, self.num_interest,1])
        if ta==3:
            loss_reg =1 * reg2 + 0.1 / reg2
            factor = tf.tile(g_factor,[1, self.num_interest,1])

        with tf.variable_scope('bilinear'):
            if self.bilinear_type == 0:
                item_emb_hat = tf.layers.dense(item_his_emb, self.dim, activation=None, bias_initializer=None)
                item_emb_hat = tf.tile(item_emb_hat, [1, 1, self.num_interest])
            elif self.bilinear_type == 1:
                item_emb_hat = tf.layers.dense(item_his_emb, self.dim * self.num_interest, activation=None, bias_initializer=None)
            else:
                w = tf.get_variable(
                    'weights', shape=[1, self.seq_len, self.num_interest * self.dim, self.dim],
                    initializer=tf.random_normal_initializer())
                # [N, T, 1, C]
                u = tf.expand_dims(item_his_emb, axis=2)
                # [N, T, num_caps * dim_caps]
                item_emb_hat = tf.reduce_sum(w[:, :self.seq_len, :, :] * u, axis=3)

        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.seq_len, self.num_interest, self.dim])
        item_emb_hat = tf.transpose(item_emb_hat, [0, 2, 1, 3])
        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.num_interest, self.seq_len, self.dim])

        if self.stop_grad:
            item_emb_hat_iter = tf.stop_gradient(item_emb_hat, name='item_emb_hat_iter')
        else:
            item_emb_hat_iter = item_emb_hat

        if self.bilinear_type > 0:
            capsule_weight = tf.stop_gradient(tf.zeros([get_shape(item_his_emb)[0], self.num_interest, self.seq_len]))
        else:
            capsule_weight = tf.stop_gradient(tf.truncated_normal([get_shape(item_his_emb)[0], self.num_interest, self.seq_len], stddev=1.0))

        for i in range(3):
            atten_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_interest, 1])
            paddings = tf.zeros_like(atten_mask)

            capsule_softmax_weight = tf.nn.softmax(capsule_weight, axis=1)
            capsule_softmax_weight = tf.where(tf.equal(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = tf.matmul(item_emb_hat_iter, tf.transpose(interest_capsule, [0, 1, 3, 2]))
                delta_weight = tf.reshape(delta_weight, [-1, self.num_interest, self.seq_len])
                if ta:
                    with tf.control_dependencies(print_ops):
                        delta_weight = delta_weight * factor
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = tf.reshape(interest_capsule, [-1, self.num_interest, self.dim])

        if self.relu_layer:
            interest_capsule = tf.layers.dense(interest_capsule, self.dim, activation=tf.nn.relu, name='proj')

        atten = tf.matmul(interest_capsule, tf.reshape(item_eb, [-1, self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [-1, self.num_interest]), 1))

        if self.hard_readout:
            readout = tf.gather(tf.reshape(interest_capsule, [-1, self.dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_his_emb)[0]) * self.num_interest)
        else:
            readout = tf.matmul(tf.reshape(atten, [get_shape(item_his_emb)[0], 1, self.num_interest]), interest_capsule)
            readout = tf.reshape(readout, [get_shape(item_his_emb)[0], self.dim])

        return interest_capsule, readout, loss_reg

class Model_MIND(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, ta=0, tmax=1000, tmin=1000, relu_layer=True, hard_readout=True):
        super(Model_MIND, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="MIND")

        item_his_emb = self.item_his_eb

        capsule_network = CapsuleNetwork(embedding_dim,hidden_size, seq_len, bilinear_type=0, num_interest=num_interest, hard_readout=hard_readout, relu_layer=relu_layer, tmax=tmax, tmin=tmin)
        self.user_eb, self.readout, self.loss_reg = capsule_network(item_his_emb, self.item_eb,  self.t_batch_ph, self.t_his_batch_ph, self.mask, ta)

        self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_reg)

class Model_ComiRec_DR(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, ta=0, tmax=1000, tmin=1000, hard_readout=True, relu_layer=False):
        super(Model_ComiRec_DR, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="ComiRec_DR")

        item_his_emb = self.item_his_eb

        capsule_network = CapsuleNetwork(embedding_dim, hidden_size, seq_len, bilinear_type=2, num_interest=num_interest, hard_readout=hard_readout, relu_layer=relu_layer, tmax=tmax, tmin=tmin)
        self.user_eb, self.readout, _ = capsule_network(item_his_emb, self.item_eb, self.t_batch_ph, self.t_his_batch_ph, self.mask, ta)

        self.build_sampled_softmax_loss(self.item_eb, self.readout, _)

class Model_ComiRec_SA(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, add_pos=True):
        super(Model_ComiRec_SA, self).__init__(n_mid, embedding_dim, hidden_size,
                                                   batch_size, seq_len, flag="ComiRec_SA")
        
        self.dim = embedding_dim
        item_list_emb = tf.reshape(self.item_his_eb, [-1, seq_len, embedding_dim])

        if add_pos:
            self.position_embedding = \
                tf.get_variable(
                    shape=[1, seq_len, embedding_dim],
                    name='position_embedding')
            item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])
        else:
            item_list_add_pos = item_list_emb

        num_heads = num_interest
        with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, hidden_size * 4, activation=tf.nn.tanh)
            item_att_w  = tf.layers.dense(item_hidden, num_heads, activation=None)
            item_att_w  = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(self.mask, axis=1), [1, num_heads, 1])
            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
            item_att_w = tf.nn.softmax(item_att_w)

            interest_emb = tf.matmul(item_att_w, item_list_emb)

        self.user_eb = interest_emb

        atten = tf.matmul(self.user_eb, tf.reshape(self.item_eb, [get_shape(item_list_emb)[0], self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [get_shape(item_list_emb)[0], num_heads]), 1))

        readout = tf.gather(tf.reshape(self.user_eb, [-1, self.dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_list_emb)[0]) * num_heads)

        self.build_sampled_softmax_loss(self.item_eb, readout)