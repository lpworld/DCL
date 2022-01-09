import tensorflow as tf

class Dual_Contrastive_Model(object):

    def __init__(self, user_count, item_count, hidden_size, batch_size, K, D):
        hidden_size = 128
        temperature = 1.0
        epsilon = 1e-20
        pruning = -10

        self.K = K
        self.D = D
        self.pruning = pruning
        self.u = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.i = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.label = tf.placeholder(tf.float32, [batch_size,]) # [B]
        self.lr = tf.placeholder(tf.float64, [])

        user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_size])
        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_size])
        user_code_w = tf.get_variable("user_code_w", [self.K*self.D, hidden_size])
        item_code_w = tf.get_variable("item_code_w", [self.K*self.D, hidden_size])
        self.user_code_w = user_code_w
        self.item_code_w = item_code_w

        item_emb = tf.nn.embedding_lookup(item_emb_w, self.i)
        user_emb = tf.nn.embedding_lookup(user_emb_w, self.u)

        # Embedding Prunning
        s = tf.Variable(self.pruning*tf.ones(1))
        item_emb = tf.sign(item_emb) * tf.nn.relu(tf.abs(item_emb)-tf.nn.sigmoid(s))
        user_emb = tf.sign(user_emb) * tf.nn.relu(tf.abs(user_emb)-tf.nn.sigmoid(s))

        # User Code Generation
        user_code = tf.layers.dense(user_emb, self.K*self.D, activation=tf.nn.tanh)
        user_code = self.gumbel_softmax(user_code, temperature, epsilon)
        #user_code = tf.log(tf.nn.softplus(user_code) + 1e-8)
        user_code = tf.reshape(user_code, [-1, self.K, self.D])
        user_code = tf.cast(tf.argmax(user_code, axis=2), tf.int32)
        self.user_code = user_code

        # Reconstructed User Embedding & Loss
        offset = tf.range(self.K, dtype="int32") * self.D
        user_reconstruct_code = user_code + offset[None, :]
        user_reconstruct_emb = tf.gather(user_code_w, user_reconstruct_code)
        user_reconstruct_emb = tf.reduce_sum(user_reconstruct_emb, axis=1)
        self.user_reconstruct_emb = user_reconstruct_emb
        self.user_reconstruction_loss = 0.5 * tf.reduce_mean(tf.reduce_sum((user_reconstruct_emb - user_emb)**2, axis=1))

        # Item Code Generation
        item_code = tf.layers.dense(item_emb, self.K*self.D, activation=tf.nn.tanh)
        item_code = self.gumbel_softmax(item_code, temperature, epsilon)
        #item_code = tf.log(tf.nn.softplus(item_code) + 1e-8)
        item_code = tf.reshape(item_code, [-1, self.K, self.D])
        item_code = tf.cast(tf.argmax(item_code, axis=2), tf.int32)
        self.item_code = item_code

        # Reconstructed Item Embedding & Loss
        offset = tf.range(self.K, dtype="int32") * self.D
        item_reconstruct_code = item_code + offset[None, :]
        item_reconstruct_emb = tf.gather(item_code_w, item_reconstruct_code)
        item_reconstruct_emb = tf.reduce_sum(item_reconstruct_emb, axis=1)
        self.item_reconstruct_emb = item_reconstruct_emb
        self.item_reconstruction_loss = 0.5 * tf.reduce_mean(tf.reduce_sum((item_reconstruct_emb - item_emb)**2, axis=1))

        # Dual Contrastive Loss
        normalize_user = tf.nn.l2_normalize(user_reconstruct_emb, 1)        
        normalize_item = tf.nn.l2_normalize(item_reconstruct_emb, 1)
        cos_similarity = tf.exp(tf.reduce_sum(tf.multiply(normalize_user,normalize_item), axis=1))
        positive = self.label * cos_similarity
        self.contrastive_loss = -tf.log(tf.reduce_sum(positive) / tf.reduce_sum(cos_similarity)+epsilon)

        # Aggregate Loss Function
        self.loss = self.user_reconstruction_loss + self.item_reconstruction_loss + 0.00001 * self.contrastive_loss

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 1)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    def gumbel_softmax(self, logits, temperature, epsilon):
        shape = tf.shape(logits)
        U = tf.random_uniform(shape, minval=0, maxval=1)
        y = logits - tf.log(-tf.log(U+epsilon)+epsilon)
        y = tf.nn.softmax(y/temperature)
        return y

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0],
                self.i: uij[1],
                self.label: uij[2],
                self.lr: lr
                })
        return loss

    def generate_code(self, sess, uij):
        user_code, item_code = sess.run([self.user_code, self.item_code], feed_dict={
                self.u: uij[0],
                self.i: uij[1],
                self.label: uij[2]
                })
        return user_code, item_code

    def save(self, sess, path='save/model'):
        saver = tf.train.Saver([self.user_code_w,self.item_code_w])
        saver.save(sess, save_path=path)

class Recommendation_Model(object):

    def __init__(self, user_count, item_count, hidden_size, batch_size, K, D):
        hidden_size = 128

        self.K = K
        self.D = D
        
        self.u = tf.placeholder(tf.int32, [batch_size,])
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.label = tf.placeholder(tf.float32, [batch_size,])
        self.user_code = tf.placeholder(tf.int32, [batch_size, self.K])
        self.item_code = tf.placeholder(tf.int32, [batch_size, self.K])
        self.lr = tf.placeholder(tf.float64, [])

        user_code_w = tf.get_variable("user_code_w", [self.K*self.D, hidden_size])
        item_code_w = tf.get_variable("item_code_w", [self.K*self.D, hidden_size])
        user_b = tf.get_variable("user_b", [user_count], initializer=tf.constant_initializer(0.0))
        item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0))

        offset = tf.range(self.K, dtype="int32") * self.D
        user_code = self.user_code + offset[None, :]
        user_code_emb = tf.gather(user_code_w, user_code)
        user_code_emb = tf.reduce_sum(user_code_emb, axis=1)

        offset = tf.range(self.K, dtype="int32") * self.D
        item_code = self.item_code + offset[None, :]
        item_code_emb = tf.gather(item_code_w, item_code)
        item_code_emb = tf.reduce_sum(item_code_emb, axis=1)

        item_b = tf.gather(item_b, self.i)
        user_b = tf.gather(user_b, self.u)

        # User-Item Feature Interaction
        concat = tf.concat([user_code_emb, item_code_emb], axis=1)
        concat = tf.layers.batch_normalization(inputs=concat)
        concat = tf.layers.dense(concat, 80, activation=tf.nn.sigmoid, name='f1')
        concat = tf.layers.dense(concat, 40, activation=tf.nn.sigmoid, name='f2')
        concat = tf.layers.dense(concat, 1, activation=None, name='f3')
        concat = tf.reshape(concat, [-1])

        # Click-Through Rate Prediction
        self.logits = item_b + concat + user_b # [B]
        self.score = tf.sigmoid(self.logits)

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=self.label))
        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 1)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0],
                self.i: uij[1],
                self.label: uij[2],
                self.user_code: uij[3],
                self.item_code: uij[4],
                self.lr: lr
                })
        return loss

    def test(self, sess, uij):
        score = sess.run(self.score, feed_dict={
                self.u: uij[0],
                self.i: uij[1],
                self.label: uij[2],
                self.user_code: uij[3],
                self.item_code: uij[4],
                })
        return score, uij[2], uij[0], uij[1]

    def seq_attention(self, inputs, hidden_size, attention_size):
        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.tile(tf.expand_dims(alphas, -1), [1, 1, hidden_size]), 1, name="attention_embedding")
        return output, alphas