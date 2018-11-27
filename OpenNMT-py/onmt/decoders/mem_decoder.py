""" Base Class and function for Decoders """

from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import onmt.models.stacked_rnn
from onmt.utils.misc import aeq
from torch.nn.init import orthogonal_, xavier_uniform_
from onmt.utils.rnn_factory import rnn_factory
from onmt.modules.memory import Memory

class MEMDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[Memory_Bank]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, num_layers,
                 hidden_size, attn_type="general", attn_func="softmax",
                 dropout=0.0, embeddings=None,
                 nr_cells=16,read_heads=4,cell_size=32,
                 independent_linears=False,share_memory=True,clip=20):
        super(MEMDecoderBase, self).__init__()
        #import pdb; pdb.set_trace()
        self.hidden_size=hidden_size
        self.num_hidden_layer=1
        self.num_layers=num_layers
        self.rnn_type=rnn_type
        self.cell_size=cell_size
        self.clip=clip
        self.read_heads=read_heads
        self.dropout=dropout
        self.embeddings = embeddings
        nn_input_size = self.embeddings.embedding_size + (cell_size*read_heads)
        nn_output_size= hidden_size + (cell_size*read_heads)
        
        self.rnns=[]
        for layer in range(self.num_layers):
            if self.rnn_type.lower() == 'rnn':
                self.rnns.append(nn.RNN((nn_input_size if layer == 0 else nn_output_size), hidden_size, dropout=dropout, batch_first=True))
            elif self.rnn_type.lower() == 'gru':
                self.rnns.append(nn.GRU((nn_input_size if layer == 0 else nn_output_size),
                                        hidden_size, dropout=dropout, batch_first=True))
            if self.rnn_type.lower() == 'lstm':
                self.rnns.append(nn.LSTM((nn_input_size if layer == 0 else nn_output_size),
                                        hidden_size, dropout=dropout, batch_first=True))
            setattr(self, rnn_type.lower() + '_layer_' + str(layer), self.rnns[layer])
        
        self.memories = []
        for layer in range(self.num_layers):
            self.memories.append(
                Memory(
                input_size=hidden_size,
                mem_size=nr_cells,
                cell_size=cell_size,
                read_heads=read_heads,
                independent_linears=independent_linears
                ).cuda()
            )
            # only one memory shared by all layers
            if share_memory:
                break
            # memories for each layer
            setattr(self, 'rnn_layer_memory_' + str(layer), self.memories[layer])
         # final output layer
        self.output = nn.Linear(nn_output_size, embeddings.embedding_size)
        orthogonal_(self.output.weight)  

        """
        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type, attn_func=attn_func
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type, attn_func=attn_func
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn
        """

    def forward(self, tgt, memory_bank, state, memory_lengths=None,
                step=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.models.DecoderState`):
                 decoder state object to initialize the decoder
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Check
        #assert isinstance(state, RNNDecoderState)
        # tgt.size() returns tgt length and batch
        _, tgt_batch, _ = tgt.size()
        #_, memory_batch, _ = memory_bank.size()
        #aeq(tgt_batch, memory_batch)
        # END
        # Run the forward pass of the RNN.
        decoder_outputs, decoder_final, attns = self._run_forward_pass(tgt, memory_bank, state, memory_lengths=memory_lengths)

        # Update the state with the result.
        final_output = decoder_outputs[-1]
        """
        coverage = None
        if "coverage" in attns:
            coverage = attns["coverage"][-1].unsqueeze(0)
        state.update_state(decoder_final, final_output.unsqueeze(0), coverage)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: decoder_outputs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(decoder_outputs) == list:
            decoder_outputs = torch.stack(decoder_outputs)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
        """
        return decoder_outputs, state, attns

    def init_decoder_state(self, src, memory_bank, encoder_final,
                           with_cache=False):
        return encoder_final

class MEMDecoder(MEMDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None,reset_experience=False, pass_through_memory=True):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                                 [len x batch x nfeats].
            memory_bank (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            decoder_final (Tensor): final hidden state from the decoder.
            decoder_outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        #import pdb; pdb.set_trace()
        emb = self.embeddings(tgt)
        max_length = emb.size(0)
        lengths    = [emb.size(0)] * max_length
        batch_size = emb.size(1)
        emb= emb.transpose(0, 1)
        controller_hidden, mem_hidden, last_read = self._init_hidden((None,state[1],None), batch_size, reset_experience)
        
        # concat xwith last read (or padding) vectors
        inputs = [torch.cat([emb[:, x, :], last_read], 1) for x in range(max_length)]
        # batched forward pass per element / word / etc
        outs = [None] * max_length
        inp=inputs[0]
        outs[0]=inp
        read_vectors = None
        # pass through time
        for time in range(1,max_length):
        # pass thorugh layers
            for layer in range(self.num_layers):
                # this layer's hidden states
                chx = controller_hidden[layer]
                m = mem_hidden if self.share_memory else mem_hidden[layer]
                # pass through controller
                outs[time], (chx, m, read_vectors) = \
                self._layer_forward(inp, layer, (chx, m), pass_through_memory)
                # store the memory back (per layer or shared)
                if self.share_memory:
                    mem_hidden = m
                else:
                    mem_hidden[layer] = m
                    controller_hidden[layer] = chx
                if read_vectors is not None:
                # the controller output + read vectors go into next layer
                    outs[time] = torch.cat([outs[time], read_vectors], 1)
                else:
                    outs[time] = torch.cat([outs[time], last_read], 1)
                inp=outs[time]
            inp=inputs[time]
            inputs[time] = outs[time]
        # pass through final output layer
        
        inputs = [self.output(i) for i in inputs]
        outputs = torch.stack(inputs)
        #if is_packed:
        #    outputs = pack(output, lengths)
        #import pdb; pdb.set_trace()
        return outputs,outputs[-1],None


    def _layer_forward(self, x, layer, hx=(None, None), pass_through_memory=True):
        (chx, mhx) = hx
        #import pdb; pdb.set_trace()
        # pass through the controller layer
        x, chx = self.rnns[layer](x.unsqueeze(1), chx)
        x = F.dropout(x.squeeze(1),self.dropout,self.training)

        # clip the controller output
        if self.clip != 0:
            output = torch.clamp(x, -self.clip, self.clip)
        else:
            output = x

        # the interface vector
        xi = output

        # pass through memory
        if pass_through_memory:
            if self.share_memory:
                read_vecs, mhx = self.memories[0](xi, mhx)
            else:
                read_vecs, mhx = self.memories[layer](xi, mhx)
            # the read vectors
            read_vectors = read_vecs.view(-1, self.cell_size * self.read_heads)
        else:
            read_vectors = None
        return output, (chx, mhx, read_vectors)
    
    def _init_hidden(self, hx, batch_size, reset_experience):
    # create empty hidden states if not provided
        if hx is None:
            hx = (None, None, None)
        (chx, mhx, last_read) = hx

        # initialize hidden state of the controller RNN
        if chx is None:
            h = torch.zeros(self.num_hidden_layer, batch_size, self.hidden_size).cuda()
            xavier_uniform_(h)
            chx = [ (h, h) if self.rnn_type.lower() == 'lstm' else h for x in range(self.num_layers)]
        # Last read vectors
        if last_read is None:
            last_read = torch.zeros(batch_size, self.cell_size * self.read_heads).cuda()
        # memory states
        if mhx is None:
            if self.share_memory:
                mhx = self.memories[0].reset(batch_size, erase=reset_experience)
            else:
                mhx = [m.reset(batch_size, erase=reset_experience) for m in self.memories]
        else:
            if self.share_memory:
                mhx = self.memories[0].reset(batch_size, mhx, erase=reset_experience)
            else:
                mhx = [m.reset(batch_size, h, erase=reset_experience) for m, h in zip(self.memories, mhx)]

        return chx, mhx, last_read
class InputFeedMEMDecoder(MEMDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[Memory_Bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = state.input_feed.squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for _, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)

            rnn_output, hidden = self.rnn(decoder_input, hidden)
            decoder_output, p_attn = self.attn(
                rnn_output,
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths)
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            decoder_outputs += [decoder_output]
            attns["std"] += [p_attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(decoder_output,
                                              memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]
        # Return result.
        return hidden, decoder_outputs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
            "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.models.stacked_rnn.StackedLSTM
        else:
            stacked_cell = onmt.models.stacked_rnn.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        """ Need to document this """
        self.hidden = tuple([_.detach() for _ in self.hidden])
        self.input_feed = self.input_feed.detach()

    def beam_update(self, idx, positions, beam_size):
        """ Need to document this """
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))

    def map_batch_fn(self, fn):
        raise NotImplementedError()


class RNNDecoderState(DecoderState):
    """ Base class for RNN decoder state """

    def __init__(self, hidden_size, rnnstate):
        """
        Args:
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = self.hidden[0].size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = self.hidden[0].data.new(*h_size).zero_() \
                              .unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        """ Update decoder state """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [e.data.repeat(1, beam_size, 1)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]

    def map_batch_fn(self, fn):
        self.hidden = tuple(map(lambda x: fn(x, 1), self.hidden))
        self.input_feed = fn(self.input_feed, 1)
