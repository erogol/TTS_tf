import tensorflow as tf
from tensorflow import keras
from TTS_tf.layers.common_layers import Prenet, Attention
from TTS_tf.utils.tf_utils import shape_list


class BatchNormConv1d(keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 stride,
                 padding="same",
                 activation=None):
        super().__init__()
        self.conv = keras.layers.Conv1D(filters, kernel_size, stride, padding)
        self.bn = keras.layers.BatchNormalization(axis=2,
                                                  momentum=0.99,
                                                  epsilon=1e-3)
        self.activation = activation

    # @tf.function(input_signature=(tf.TensorSpec(shape=[None],
                                                # dtype=tf.float32), ))
    def call(self, x):
        """
        Shapes:
            - x: B x T_in x C_in
            - o: B x T_out x C_out
        """
        o = self.bn(self.conv(x))
        if self.activation is None:
            return o
        return self.activation(o)


class Highway(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.H = keras.layers.Dense(units, bias_initializer='zeros')
        self.T = keras.layers.Dense(
            units, bias_initializer=keras.initializers.Constant(value=-1))
        self.relu = keras.layers.ReLU()
        self.sigmoid = keras.activations.sigmoid
        # self.init_layers()

    # @tf.function(input_signature=(tf.TensorSpec(shape=[None],
                                                # dtype=tf.float32), ))
    def call(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class CBHG(keras.layers.Layer):
    def __init__(self,
                 K=16,
                 conv_bank_filters=128,
                 conv_filters=[128, 128],
                 highway_units=128,
                 gru_units=128,
                 num_highways=4):
        super().__init__()
        self.K = K
        self.conv_bank_filters = conv_bank_filters
        self.conv_filters = conv_filters
        self.highway_units = highway_units
        self.gru_units = gru_units
        self.num_highways = num_highways
        self.relu = keras.layers.ReLU()

        self.conv1d_banks = [
            BatchNormConv1d(conv_bank_filters,
                            kernel_size=k,
                            stride=1,
                            padding="same",
                            activation=self.relu) for k in range(1, K + 1)
        ]

        out_features = [K * conv_bank_filters] + conv_filters[:-1]
        activations = [self.relu] * (len(conv_filters) - 1)
        activations += [None]

        self.conv1d_proj_layers = []
        for (in_size, out_size, ac) in zip(out_features, conv_filters,
                                           activations):
            layer = BatchNormConv1d(out_size,
                                    kernel_size=3,
                                    stride=1,
                                    padding="same",
                                    activation=ac)
            self.conv1d_proj_layers.append(layer)

        # setup Highway layers
        if self.highway_units != conv_filters[-1]:
            self.pre_highway = keras.layers.Dense(highway_units,
                                                  use_bias=False)
        self.highways = [Highway(highway_units) for _ in range(num_highways)]
        # bi-directional GRU layer
        self.gru = keras.layers.Bidirectional(
            keras.layers.GRU(gru_units, return_sequences=True))

    # @tf.function(input_signature=(tf.TensorSpec(shape=[None],
                                                # dtype=tf.float32), ))
    def call(self, x):
        """
        Shapes:
            - x: B x T_in x C_in
            - outputs: B x T_in x C_in * 2
        """
        res = x
        # Needed to perform conv1d on time-axis
        outs = []
        for conv1d in self.conv1d_banks:
            out = conv1d(x)
            outs.append(out)
        x = tf.concat(outs, axis=2)
        assert x.shape[2] == self.conv_bank_filters * len(self.conv1d_banks)
        for conv1d in self.conv1d_proj_layers:
            x = conv1d(x)
        # Back to the original shape
        x += res
        if self.highway_units != self.conv_filters[-1]:
            x = self.pre_highway(x)
        # Residual connection
        for highway in self.highways:
            x = highway(x)
        outputs = self.gru(x)
        return outputs


class Encoder(keras.layers.Layer):
    r"""Encapsulate Prenet and CBHG modules for encoder"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.prenet = Prenet(prenet_type="original",
                             prenet_dropout=True,
                             units=[256, 128])
        self.cbhg = CBHG(K=16,
                         conv_bank_filters=128,
                         conv_filters=[128, 128],
                         highway_units=128,
                         gru_units=128,
                         num_highways=4)

    # # @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 256],
    #                                             dtype=tf.float32), ))
    def call(self, inputs):
        r"""
        Shapes:
            - inputs: batch x time x in_features
            - outputs: batch x time x 128*2
        """
        inputs = self.prenet(inputs)
        return self.cbhg(inputs)


class Postnet(keras.layers.Layer):
    def __init__(self, mel_dim):
        super().__init__()
        self.cbhg = CBHG(K=8,
                         conv_bank_filters=128,
                         conv_filters=[256, mel_dim],
                         highway_units=128,
                         gru_units=128,
                         num_highways=4)

    # @tf.function(input_signature=(tf.TensorSpec(shape=[None],
    #                                             dtype=tf.float32), ))
    def call(self, x):
        return self.cbhg(x)


class Decoder(keras.layers.Layer):
    """Decoder module.

    Args:
        in_features (int): input vector (encoder output) sample size.
        memory_dim (int): memory vector (prev. time-step output) sample size.
        r (int): number of outputs per time step.
        memory_size (int): size of the past window. if <= 0 memory_size = r
        TODO: arguments
    """

    # Pylint gets confused by PyTorch conventions here
    #pylint: disable=attribute-defined-outside-init

    def __init__(self, input_dim, memory_dim, r, memory_size,
                 use_attn_windowing, attn_norm, prenet_type, prenet_dropout,
                 use_forward_attn, use_trans_agent, use_forward_attn_mask,
                 use_loc_attn, separate_stopnet, speaker_embedding_dim):
        super(Decoder, self).__init__()
        self.r_init = r
        self.r = r
        self.input_dim = input_dim
        self.max_decoder_steps = 500
        self.use_memory_queue = memory_size > 0
        self.memory_size = memory_size if memory_size > 0 else r
        self.memory_dim = memory_dim
        self.separate_stopnet = separate_stopnet
        self.query_dim = 256
        self.attn_dim = 128
        # memory -> |Prenet| -> processed_memory
        self.prenet = Prenet(prenet_type, prenet_dropout, units=[256, 128])
        # processed_inputs, processed_memory -> |Attention| -> Attention, attention, RNN_State
        # attention_rnn generates queries for the attention mechanism
        self.attention_rnn = keras.layers.GRU(self.query_dim)

        self.attention = Attention(attn_dim=self.attn_dim,
                                   use_loc_attn=use_loc_attn,
                                   loc_attn_n_filters=32,
                                   loc_attn_kernel_size=31,
                                   use_windowing=use_attn_windowing,
                                   norm=attn_norm,
                                   use_forward_attn=use_forward_attn,
                                   use_trans_agent=use_trans_agent,
                                   use_forward_attn_mask=use_forward_attn_mask)
        # (processed_memory | attention context) -> |Linear| -> decoder_RNN_input
        self.project_to_decoder_in = keras.layers.Dense(256)
        # decoder_RNN_input -> |RNN| -> RNN_state
        self.decoder_rnns = [keras.layers.GRU(256) for _ in range(2)]
        # RNN_state -> |Linear| -> mel_spec
        self.proj_to_mel = keras.layers.Dense(memory_dim * self.r_init)
        # learn init values instead of zero init.
        self.stopnet = StopNet(256 + memory_dim * self.r_init)

    def set_r(self, new_r):
        self.r = new_r

    def _reshape_memory(self, memory):
        """
        Reshape the spectrograms for given 'r'
        """
        B = memory.shape[0]
        # Grouping multiple frames if necessary
        if memory.shape[-1] == self.memory_dim and self.r > 1:
            memory = tf.reshape(memory, [B, -1, tf.shape(memory)[2] * self.r])
        # Time first (T_decoder, B, memory_dim)
        memory = tf.transpose(memory, perm=[1, 0, 2])
        return memory

    def _init_states(self, inputs):
        """
        Initialization of decoder states
        """
        B = shape_list(inputs)[0]
        T = shape_list(inputs)[1]
        self.attention.init_states(inputs)
        # go frame as zeros matrix
        if self.use_memory_queue:
            self.memory_input = tf.zeros(
                [B, self.memory_dim * self.memory_size])
        else:
            self.memory_input = tf.zeros([B, self.memory_dim])
        # decoder states
        self.attention_rnn_hidden = tf.zeros([B, 256])
        self.decoder_rnn_hiddens = [
            tf.zeros([B, 256]) for idx in range(len(self.decoder_rnns))
        ]
        self.context_vec = tf.zeros([B, 1, self.input_dim])
        self.attention.process_values(inputs)

    def _parse_outputs(self, outputs, attentions, stop_tokens):
        # Back to batch first
        attentions = tf.transpose(tf.stack(attentions), (1, 0, 2))
        outputs = tf.transpose(tf.stack(outputs), (1, 0, 2))
        outputs = tf.reshape(outputs, [outputs.shape[0], -1, self.memory_dim])
        stop_tokens = tf.squeeze(
            tf.transpose(tf.stack(stop_tokens), (1, 0, 2)), 2)
        return outputs, attentions, stop_tokens

    def decode(self, inputs, processed_memory, mask=None):
        # Attention RNN
        attention_rnn_input = tf.concat([processed_memory, self.context_vec], -1)
        self.attention_rnn_hidden = self.attention_rnn(attention_rnn_input, initial_state=self.attention_rnn_hidden)
        self.context_vec = self.attention(self.attention_rnn_hidden, inputs,
                                          mask)
        # Concat RNN output and attention context vector
        decoder_input = self.project_to_decoder_in(
            tf.concat([tf.expand_dims(self.attention_rnn_hidden, axis=1), self.context_vec], -1))
        # Pass through the decoder RNNs
        for idx in range(len(self.decoder_rnns)):
            self.decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](
                decoder_input, initial_state=self.decoder_rnn_hiddens[idx])
            # Residual connection
            decoder_input = tf.expand_dims(self.decoder_rnn_hiddens[idx],
                                           1) + decoder_input
        decoder_output = tf.squeeze(decoder_input, axis=1)

        # predict mel vectors from decoder vectors
        output = self.proj_to_mel(decoder_output)
        # output = torch.sigmoid(output)
        # predict stop token
        stopnet_input = tf.concat([decoder_output, output], -1)
        if self.separate_stopnet:
            stop_token = self.stopnet(tf.stop_gradient(stopnet_input))
        else:
            stop_token = self.stopnet(stopnet_input)
        output = output[:, :self.r * self.memory_dim]
        return output, stop_token, tf.squeeze(self.attention.attn_weights, -1)

    def _update_memory_input(self, new_memory):
        self.memory_input = new_memory[:, self.memory_dim * (self.r - 1):]

    def call(self, inputs, memory, mask):
        """
        Args:
            inputs: Encoder outputs.`
            memory: Decoder memory (autoregression. If None (at eval-time),
              decoder outputs are used as decoder inputs. If None, it uses the last
              output as the input.
            mask: Attention mask for sequence padding.

        Shapes:
            - inputs: batch x time x encoder_out_dim
            - memory: batch x #mel_specs x mel_spec_dim
        """
        # Run greedy decoding if memory is None
        # memory = self._reshape_memory(memory)
        B, T, D = shape_list(memory)
        num_iter = shape_list(memory)[1] // self.r
        memory_zero = tf.zeros([B, 1, D])
        outputs = tf.TensorArray(dtype=tf.float32, size=num_iter)
        attentions = tf.TensorArray(dtype=tf.float32, size=num_iter)
        stop_tokens = tf.TensorArray(dtype=tf.float32, size=num_iter)
        self._init_states(inputs)
            #stop_tokens.append(stop_tokenr
        memory_aligned = memory[:, (self.r - 1):(T-self.r):self.r]
        prenet_in = tf.concat([memory_zero, memory_aligned], axis=1)
        prenet_out = self.prenet(prenet_in)
        # TODO: memory queuing
        t = 0
        while t < num_iter:
            new_memory = tf.expand_dims(prenet_out[:, t], axis=1)
            # if speaker_embeddings is not None:
            # self.memory_input = tf.concat([self.memory_input, speaker_embeddings], axis=-1)
            output, stop_token, attention = self.decode(inputs, new_memory, mask)
            outputs = outputs.write(t, output)
            attentions = attentions.write(t, attention)
            stop_tokens = stop_tokens.write(t, stop_token)
            t += 1
        outputs = outputs.stack()
        attentions = attentions.stack()
        stop_tokens = stop_tokens.stack()
        outputs = tf.transpose(outputs, [1, 0, 2])
        attentions = tf.transpose(attentions, [1, 0 ,2])
        stop_tokens = tf.transpose(stop_tokens, [1, 0, 2])
        stop_tokens = tf.squeeze(stop_tokens, axis=2)
        return outputs, attentions, stop_tokens

    def inference(self, inputs, speaker_embeddings=None):
        """
        Args:
            inputs: encoder outputs.
            speaker_embeddings: speaker vectors.

        Shapes:
            - inputs: batch x time x encoder_out_dim
            - speaker_embeddings: batch x embed_dim
        """
        outputs = []
        attentions = []
        stop_tokens = []
        t = 0
        self._init_states(inputs)
        while True:
            if t > 0:
                new_memory = outputs[-1]
                self._update_memory_input(new_memory)
            if speaker_embeddings is not None:
                self.memory_input = tf.concat(
                    [self.memory_input, speaker_embeddings], dim=-1)
            output, stop_token, attention = self.decode(inputs, None)
            stop_token = tf.nn.sigmoid(stop_token)
            outputs += [output]
            attentions += [attention]
            stop_tokens += [stop_token]
            t += 1
            if t > inputs.shape[1] / 4 and (stop_token > 0.6
                                            or attention[:, -1] > 0.6):
                break
            elif t > self.max_decoder_steps:
                print("   | > Decoder stopped with 'max_decoder_steps")
                break
        return self._parse_outputs(outputs, attentions, stop_tokens)


class StopNet(keras.layers.Layer):
    r"""
    Args:
        in_features (int): feature dimension of input.
    """
    def __init__(self, in_features):
        super(StopNet, self).__init__()
        self.dropout = keras.layers.Dropout(rate=0.1)
        self.linear = keras.layers.Dense(1)

    # @tf.function
    def call(self, inputs):
        outputs = self.dropout(inputs)
        outputs = self.linear(outputs)
        return outputs
