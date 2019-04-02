"""define Factorization-Machine based Neural Network Model"""
import math
import numpy as np
import tensorflow as tf
from src.base_model import BaseModel

__all__ = ["ExtremeDeepFMModel"]


class ExtremeDeepFMModel(BaseModel):
    """define Factorization-Machine based Neural Network Model"""

    def _build_graph(self, hparams):
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)
        self.layer_keeps = tf.placeholder(tf.float32)
        with tf.variable_scope("exDeepFm") as scope:
            with tf.variable_scope("embedding", initializer=self.initializer) as escope:
                self.embedding = tf.get_variable(name='embedding_layer',
                                                 shape=[hparams.FEATURE_COUNT, hparams.dim],
                                                 dtype=tf.float32)
                self.embed_params.append(self.embedding)
                embed_out, embed_layer_size = self._build_embedding(hparams)
            logit = self._build_linear(hparams)
            # logit = tf.add(logit, self._build_fm(hparams))
            # res: use resnet?  direct: without split?  reduce_D: Dimension reduction?  f_dim: dimension of reduce_D
            logit = tf.add(logit, self._build_extreme_FM(hparams, embed_out, res=False, direct=False, bias=False, reduce_D=False, f_dim=2))
            # logit = tf.add(logit, self._build_extreme_FM_quick(hparams, embed_out))
            logit = tf.add(logit, self._build_dnn(hparams, embed_out, embed_layer_size))
            return logit

    def _build_embedding(self, hparams):
        fm_sparse_index = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                          self.iterator.dnn_feat_values,
                                          self.iterator.dnn_feat_shape)
        fm_sparse_weight = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                           self.iterator.dnn_feat_weights,
                                           self.iterator.dnn_feat_shape)
        w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse(self.embedding,
                                                            fm_sparse_index,
                                                            fm_sparse_weight,
                                                            combiner="sum")
        embedding = tf.reshape(w_fm_nn_input_orgin, [-1, hparams.dim * hparams.FIELD_COUNT])
        embedding_size = hparams.FIELD_COUNT * hparams.dim
        return embedding, embedding_size

    def _build_linear(self, hparams):
        with tf.variable_scope("linear_part", initializer=self.initializer) as scope:
            w_linear = tf.get_variable(name='w',
                                       shape=[hparams.FEATURE_COUNT, 1],
                                       dtype=tf.float32)
            b_linear = tf.get_variable(name='b',
                                       shape=[1],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer())
            x = tf.SparseTensor(self.iterator.fm_feat_indices,
                                self.iterator.fm_feat_values,
                                self.iterator.fm_feat_shape)
            linear_output = tf.add(tf.sparse_tensor_dense_matmul(x, w_linear), b_linear)
            self.layer_params.append(w_linear)
            self.layer_params.append(b_linear)
            tf.summary.histogram("linear_part/w", w_linear)
            tf.summary.histogram("linear_part/b", b_linear)
            return linear_output

    def _build_fm(self, hparams):
        with tf.variable_scope("fm_part") as scope:
            x = tf.SparseTensor(self.iterator.fm_feat_indices,
                                self.iterator.fm_feat_values,
                                self.iterator.fm_feat_shape)
            xx = tf.SparseTensor(self.iterator.fm_feat_indices,
                                 tf.pow(self.iterator.fm_feat_values, 2),
                                 self.iterator.fm_feat_shape)
            fm_output = 0.5 * tf.reduce_sum(
                tf.pow(tf.sparse_tensor_dense_matmul(x, self.embedding), 2) - \
                tf.sparse_tensor_dense_matmul(xx,
                                              tf.pow(self.embedding, 2)), 1,
                keep_dims=True)
            return fm_output

    def _build_extreme_FM(self, hparams, nn_input, res=False, direct=False, bias=False, reduce_D=False, f_dim=2):
        hidden_nn_layers = []
        field_nums = []
        final_len = 0
        field_num = hparams.FIELD_COUNT  # 33
        nn_input = tf.reshape(nn_input, shape=[-1, int(field_num), hparams.dim]) # (?, 330) => (?, 33, 10)
        field_nums.append(int(field_num))
        hidden_nn_layers.append(nn_input)
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0], hparams.dim * [1], 2) # 切分成list  (?, 33, 1) * 10
        with tf.variable_scope("exfm_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.cross_layer_sizes): # [100, 100, 50]
                print('idx %d, layer_size %d'%(idx, layer_size))
                split_tensor = tf.split(hidden_nn_layers[-1], hparams.dim * [1], 2)
                dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)  # (?, 33, 1) * 10  第一层 (?, 33, 1) * 10  第二层  (?, 50, 1) * 10
                                                                                         #
                                                                                         #
                dot_result_o = tf.reshape(dot_result_m, shape=[hparams.dim, -1, field_nums[0]*field_nums[-1]])
                dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

######          idx 0, layer_size 100
#               hidden_nn_layers[-1] Tensor("exDeepFm/Reshape:0", shape=(?, 33, 10), dtype=float32)
#               split_tensor         10 [<tf.Tensor 'exDeepFm/exfm_part/split:0' shape=(?, 33, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split:1' shape=(?, 33, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split:2' shape=(?, 33, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split:3' shape=(?, 33, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split:4' shape=(?, 33, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split:5' shape=(?, 33, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split:6' shape=(?, 33, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split:7' shape=(?, 33, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split:8' shape=(?, 33, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split:9' shape=(?, 33, 1) dtype=float32>]
#               dot_result_m         Tensor("exDeepFm/exfm_part/MatMul:0", shape=(10, ?, 33, 33), dtype=float32)
#               dot_result_o         Tensor("exDeepFm/exfm_part/Reshape:0", shape=(10, ?, 1089), dtype=float32)
#               dot_result           Tensor("exDeepFm/exfm_part/transpose:0", shape=(?, 10, 1089), dtype=float32)  1089 = 33*33  33*33个10维向量
#                                      关键点：同层不同vector的区别仅仅在于不同的加和权重矩阵，dot_result是提前计算好两两向量间Hadamard乘的结果 
######          idx 1, layer_size 100
#               hidden_nn_layers[-1] Tensor("exDeepFm/exfm_part/split_1:0", shape=(?, 50, 10), dtype=float32)
#               split_tensor         10 [<tf.Tensor 'exDeepFm/exfm_part/split_2:0' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_2:1' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_2:2' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_2:3' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_2:4' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_2:5' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_2:6' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_2:7' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_2:8' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_2:9' shape=(?, 50, 1) dtype=float32>]
#               dot_result_m         Tensor("exDeepFm/exfm_part/MatMul_1:0", shape=(10, ?, 33, 50), dtype=float32)
#               dot_result_o         Tensor("exDeepFm/exfm_part/Reshape_1:0", shape=(10, ?, 1650), dtype=float32)
#               dot_result           Tensor("exDeepFm/exfm_part/transpose_2:0", shape=(?, 10, 1650), dtype=float32)  33*50个10维向量
######          idx 2, layer_size 50
#               hidden_nn_layers[-1] Tensor("exDeepFm/exfm_part/split_3:0", shape=(?, 50, 10), dtype=float32)
#               split_tensor         10 [<tf.Tensor 'exDeepFm/exfm_part/split_4:0' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_4:1' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_4:2' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_4:3' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_4:4' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_4:5' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_4:6' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_4:7' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_4:8' shape=(?, 50, 1) dtype=float32>, <tf.Tensor 'exDeepFm/exfm_part/split_4:9' shape=(?, 50, 1) dtype=float32>]
#               dot_result_m         Tensor("exDeepFm/exfm_part/MatMul_2:0", shape=(10, ?, 33, 50), dtype=float32)
#               dot_result_o         Tensor("exDeepFm/exfm_part/Reshape_2:0", shape=(10, ?, 1650), dtype=float32)
#               dot_result           Tensor("exDeepFm/exfm_part/transpose_4:0", shape=(?, 10, 1650), dtype=float32)

                if reduce_D: # False
                    hparams.logger.info("reduce_D")
                    filters0 = tf.get_variable("f0_" + str(idx),
                                               shape=[1, layer_size, field_nums[0], f_dim],
                                               dtype=tf.float32)
                    filters_ = tf.get_variable("f__" + str(idx),
                                               shape=[1, layer_size, f_dim, field_nums[-1]],
                                               dtype=tf.float32)
                    filters_m = tf.matmul(filters0, filters_)
                    filters_o = tf.reshape(filters_m, shape=[1, layer_size, field_nums[0] * field_nums[-1]])
                    filters = tf.transpose(filters_o, perm=[0, 2, 1])
                else:
                    filters = tf.get_variable(name="f_"+str(idx),
                                         shape=[1, field_nums[-1]*field_nums[0], layer_size],
                                         dtype=tf.float32)
                # dot_result = tf.transpose(dot_result, perm=[0, 2, 1])
                # 一维卷积只在width或者说height方向上进行滑窗并相乘求和
                curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')

                # BIAS ADD False
                if bias:
                    hparams.logger.info("bias")
                    b = tf.get_variable(name="f_b" + str(idx),
                                    shape=[layer_size],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                    curr_out = tf.nn.bias_add(curr_out, b)
                    self.cross_params.append(b)
                    self.layer_params.append(b)

                curr_out = self._activate(curr_out, hparams.cross_activation)
                curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

#######         idx 0, layer_size 100
#               dot_result             Tensor("exDeepFm/exfm_part/transpose:0", shape=(?, 10, 1089), dtype=float32)     33*33个10维向量
#               filters                <tf.Variable 'exDeepFm/exfm_part/f_0:0' shape=(1, 1089, 100) dtype=float32_ref>  100个权重矩阵
#               curr_out(包括activate) Tensor("exDeepFm/exfm_part/conv1d/Squeeze:0", shape=(?, 10, 100), dtype=float32)
#               transpose curr_out     Tensor("exDeepFm/exfm_part/transpose_1:0", shape=(?, 100, 10), dtype=float32)
#               next_hidden            Tensor("exDeepFm/exfm_part/split_1:0", shape=(?, 50, 10), dtype=float32)
#               direct_connect         Tensor("exDeepFm/exfm_part/split_1:1", shape=(?, 50, 10), dtype=float32)
#######         idx 1, layer_size 100
#               dot_result             Tensor("exDeepFm/exfm_part/transpose_2:0", shape=(?, 10, 1650), dtype=float32)
#               filters                <tf.Variable 'exDeepFm/exfm_part/f_1:0' shape=(1, 1650, 100) dtype=float32_ref>
#               curr_out(包括activate) Tensor("exDeepFm/exfm_part/conv1d_1/Squeeze:0", shape=(?, 10, 100), dtype=float32)
#               transpose curr_out     Tensor("exDeepFm/exfm_part/transpose_3:0", shape=(?, 100, 10), dtype=float32)
#               next_hidden            Tensor("exDeepFm/exfm_part/split_1:0", shape=(?, 50, 10), dtype=float32)
#               direct_connect         Tensor("exDeepFm/exfm_part/split_1:1", shape=(?, 50, 10), dtype=float32)
#######         idx 2, layer_size 50
#               dot_result             Tensor("exDeepFm/exfm_part/transpose_4:0", shape=(?, 10, 1650), dtype=float32)
#               filters                <tf.Variable 'exDeepFm/exfm_part/f_2:0' shape=(1, 1650, 50) dtype=float32_ref>
#               curr_out(包括activate) Tensor("exDeepFm/exfm_part/conv1d_2/Squeeze:0", shape=(?, 10, 50), dtype=float32)
#               transpose curr_out     Tensor("exDeepFm/exfm_part/transpose_5:0", shape=(?, 50, 10), dtype=float32)
#               direct_connect         Tensor("exDeepFm/exfm_part/transpose_5:0", shape=(?, 50, 10), dtype=float32)

                if direct: #False
                    hparams.logger.info("all direct connect")
                    direct_connect = curr_out
                    next_hidden = curr_out
                    final_len += layer_size
                    field_nums.append(int(layer_size))
                else:
                    hparams.logger.info("split connect")
                    if idx != len(hparams.cross_layer_sizes) - 1:
                        next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                        final_len += int(layer_size / 2)
                    else:
                        direct_connect = curr_out
                        next_hidden = 0
                        final_len += layer_size
                    field_nums.append(int(layer_size / 2))

                final_result.append(direct_connect)
                hidden_nn_layers.append(next_hidden)

                self.cross_params.append(filters)
                self.layer_params.append(filters)

            result = tf.concat(final_result, axis=1)  # Tensor("exDeepFm/exfm_part/concat:0", shape=(?, 150, 10), dtype=float32)
            result = tf.reduce_sum(result, -1)        # Tensor("exDeepFm/exfm_part/Sum:0", shape=(?, 150), dtype=float32)
            if res: #False
                hparams.logger.info("residual network")
                w_nn_output1 = tf.get_variable(name='w_nn_output1',
                                               shape=[final_len, 128],
                                               dtype=tf.float32)
                b_nn_output1 = tf.get_variable(name='b_nn_output1',
                                               shape=[128],
                                               dtype=tf.float32,
                                               initializer=tf.zeros_initializer())
                self.layer_params.append(w_nn_output1)
                self.layer_params.append(b_nn_output1)
                exFM_out0 = tf.nn.xw_plus_b(result, w_nn_output1, b_nn_output1)
                exFM_out1 = self._active_layer(logit=exFM_out0,
                                               scope=scope,
                                               activation="relu",
                                               layer_idx=0)
                w_nn_output2 = tf.get_variable(name='w_nn_output2',
                                               shape=[128 + final_len, 1],
                                               dtype=tf.float32)
                b_nn_output2 = tf.get_variable(name='b_nn_output2',
                                               shape=[1],
                                               dtype=tf.float32,
                                               initializer=tf.zeros_initializer())
                self.layer_params.append(w_nn_output2)
                self.layer_params.append(b_nn_output2)
                exFM_in = tf.concat([exFM_out1, result], axis=1, name="user_emb")
                exFM_out = tf.nn.xw_plus_b(exFM_in, w_nn_output2, b_nn_output2)

            else:
                hparams.logger.info("no residual network")
                w_nn_output = tf.get_variable(name='w_nn_output',
                                              shape=[final_len, 1],
                                              dtype=tf.float32)
                b_nn_output = tf.get_variable(name='b_nn_output',
                                              shape=[1],
                                              dtype=tf.float32,
                                              initializer=tf.zeros_initializer())
                self.layer_params.append(w_nn_output)
                self.layer_params.append(b_nn_output)
                exFM_out = tf.nn.xw_plus_b(result, w_nn_output, b_nn_output)

#            total_parameters = 0
#            for variable in tf.trainable_variables():
#                shape = variable.get_shape()
#                variable_parameters = 1
#                for dim in shape:
#                    variable_parameters *= dim.value
#                total_parameters += variable_parameters
#                if len(shape) == 1:
#                    print("%s %d, " % (variable.name, variable_parameters))
#                else:
#                    print("%s %s=%d, " % (variable.name, str(shape), variable_parameters))
#            print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))

#exDeepFm/embedding/embedding_layer:0 (194081, 10)=1940810,
#exDeepFm/linear_part/w:0 (194081, 1)=194081,
#exDeepFm/linear_part/b:0 1,
#exDeepFm/exfm_part/f_0:0 (1, 1089, 100)=108900,
#exDeepFm/exfm_part/f_1:0 (1, 1650, 100)=165000,
#exDeepFm/exfm_part/f_2:0 (1, 1650, 50)=82500,
#exDeepFm/exfm_part/w_nn_output:0 (150, 1)=150,
#exDeepFm/exfm_part/b_nn_output:0 1,
#Total 8 variables, 2,491,443 params

            return exFM_out

    def _build_extreme_FM_quick(self, hparams, nn_input):
        hidden_nn_layers = []
        field_nums = []
        final_len = 0
        field_num = hparams.FIELD_COUNT
        nn_input = tf.reshape(nn_input, shape=[-1, int(field_num), hparams.dim])
        field_nums.append(int(field_num))
        hidden_nn_layers.append(nn_input)
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0], hparams.dim * [1], 2)
        with tf.variable_scope("exfm_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.cross_layer_sizes):
                split_tensor = tf.split(hidden_nn_layers[-1], hparams.dim * [1], 2)
                dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
                dot_result_o = tf.reshape(dot_result_m, shape=[hparams.dim, -1, field_nums[0]*field_nums[-1]])
                dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

                filters = tf.get_variable(name="f_"+str(idx),
                                         shape=[1, field_nums[-1]*field_nums[0], layer_size],
                                         dtype=tf.float32)
                # dot_result = tf.transpose(dot_result, perm=[0, 2, 1])
                curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')


                curr_out = tf.transpose(curr_out, perm=[0, 2, 1])


                hparams.logger.info("split connect")
                if idx != len(hparams.cross_layer_sizes) - 1:
                    next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                    final_len += int(layer_size / 2)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
                    final_len += layer_size
                field_nums.append(int(layer_size / 2))

                final_result.append(direct_connect)
                hidden_nn_layers.append(next_hidden)

                self.cross_params.append(filters)

            result = tf.concat(final_result, axis=1)
            result = tf.reduce_sum(result, -1)

            hparams.logger.info("no residual network")
            w_nn_output = tf.get_variable(name='w_nn_output',
                                              shape=[final_len, 1],
                                              dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                              shape=[1],
                                              dtype=tf.float32,
                                              initializer=tf.zeros_initializer())
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            exFM_out = tf.nn.xw_plus_b(result, w_nn_output, b_nn_output)

            return exFM_out


    def _build_dnn(self, hparams, embed_out, embed_layer_size):
        """
        fm_sparse_index = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                          self.iterator.dnn_feat_values,
                                          self.iterator.dnn_feat_shape)
        fm_sparse_weight = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                           self.iterator.dnn_feat_weights,
                                           self.iterator.dnn_feat_shape)
        w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse(self.embedding,
                                                            fm_sparse_index,
                                                            fm_sparse_weight,
                                                            combiner="sum")
        w_fm_nn_input = tf.reshape(w_fm_nn_input_orgin, [-1, hparams.dim * hparams.FIELD_COUNT])
        last_layer_size = hparams.FIELD_COUNT * hparams.dim
        """
        w_fm_nn_input = embed_out
        last_layer_size = embed_layer_size
        layer_idx = 0
        hidden_nn_layers = []
        hidden_nn_layers.append(w_fm_nn_input)
        with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.layer_sizes):
                curr_w_nn_layer = tf.get_variable(name='w_nn_layer' + str(layer_idx),
                                                  shape=[last_layer_size, layer_size],
                                                  dtype=tf.float32)
                curr_b_nn_layer = tf.get_variable(name='b_nn_layer' + str(layer_idx),
                                                  shape=[layer_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer())
                tf.summary.histogram("nn_part/" + 'w_nn_layer' + str(layer_idx),
                                     curr_w_nn_layer)
                tf.summary.histogram("nn_part/" + 'b_nn_layer' + str(layer_idx),
                                     curr_b_nn_layer)
                curr_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[layer_idx],
                                                       curr_w_nn_layer,
                                                       curr_b_nn_layer)
                scope = "nn_part" + str(idx)
                activation = hparams.activation[idx]
                curr_hidden_nn_layer = self._active_layer(logit=curr_hidden_nn_layer,
                                                          scope=scope,
                                                          activation=activation,
                                                          layer_idx=idx)
                hidden_nn_layers.append(curr_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size
                self.layer_params.append(curr_w_nn_layer)
                self.layer_params.append(curr_b_nn_layer)

            w_nn_output = tf.get_variable(name='w_nn_output',
                                          shape=[last_layer_size, 1],
                                          dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                          shape=[1],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
            tf.summary.histogram("nn_part/" + 'w_nn_output' + str(layer_idx),
                                 w_nn_output)
            tf.summary.histogram("nn_part/" + 'b_nn_output' + str(layer_idx),
                                 b_nn_output)
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            nn_output = tf.nn.xw_plus_b(hidden_nn_layers[-1], w_nn_output, b_nn_output)
            return nn_output

#https://www.jianshu.com/p/b4128bc79df0
#import tensorflow as tf
#import numpy as np
#
#arr1 = tf.convert_to_tensor(np.arange(1,25).reshape(2,4,3),dtype=tf.int32)
#arr2 = tf.convert_to_tensor(np.arange(1,25).reshape(2,4,3),dtype=tf.int32)
#
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    split_arr1 = tf.split(arr1,[1,1,1],2) # 3 elements [<tf.Tensor 'split_2:0' shape=(2, 4, 1) dtype=int32>, <tf.Tensor 'split_2:1' shape=(2, 4, 1) dtype=int32>, <tf.Tensor 'split_2:2' shape=(2, 4, 1) dtype=int32>]
#    split_arr2 = tf.split(arr2,[1,1,1],2)
#    print(split_arr1)
#    print(sess.run(split_arr1))
#    print(sess.run(split_arr2))
#    print(len(split_arr1))     # 3 split_arr1是一个list
#    print(split_arr1[0].shape) # (2, 4, 1)
#    res = tf.matmul(split_arr1,split_arr2,transpose_b=True)
#    print(sess.run(res))
#    print(res.shape)  #(3, 2, 4, 4)
#    res = tf.transpose(res,perm=[1,0,2,3])
#    print(sess.run(res))
#    print(res.shape)  #(2, 3, 4, 4)
#
#split_arr1:
#[<tf.Tensor 'split_2:0' shape=(2, 4, 1) dtype=int32>, <tf.Tensor 'split_2:1' shape=(2, 4, 1) dtype=int32>, <tf.Tensor 'split_2:2' shape=(2, 4, 1) dtype=int32>]
#[array([[[ 1],[ 4],[ 7],[10]],
#        [[13],[16],[19],[22]]]), 
# array([[[ 2],[ 5],[ 8],[11]],
#       [[14],[17],[20],[23]]]), 
# array([[[ 3],[ 6],[ 9],[12]],
#       [[15],[18],[21],[24]]])]

