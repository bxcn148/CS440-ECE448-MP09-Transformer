Download link :https://programming.engineering/product/cs440-ece448-mp09-transformer/

# CS440-ECE448-MP09-Transformer
CS440/ECE448 MP09: Transformer
The first thing you need to do is to download mp09.zip. Unlike several previous MPs, you will need to complete the code in several .py files

mp09_notebook.ipynb

This file ( ) will walk you through the whole MP, giving you instructions and debugging tips as you go.

Throughout this MP, you will implement many of the operations in the Transformer

architecture. The implementation will mostly follow the instructions in the original paper

“Attention is All You Need” by Vaswani et al, 2017. (https://arxiv.org/abs/1706.03762). Much of the class structures and code have already been completed for you, and you are only required to fill in the missing parts as instructed by the comments in the code and in this notebook.

Table of Contents



Code Structure



Multi-head Attention



Positional Encoding



Transformer Encoder



Transformer Decoder



Extra Credit: Auto-regressive Decoding During Inference

Code structure

The repository, once unzipped, is structured like this:

├─ data – Folder that contains either synthetic or real data and the expected outputs, for use by the tests.

│

├─ tests – Folder that contains tests used by grade.py

│

├─ mha.py – Implements multi-head attention; you need to implement some part of it

│

├─ pe.py – Implements positional encoding; you need to implement some part of it

│

├─ encoder.py – Implements the Transformer encoder layer and the encoder; you need to implement some part of it

│

4/5/24, 5:26 PM mp09_notebook

├─ decoder.py – Implements the Transformer decoder layer and the decoder; you need to implement some part of it.

│

├─ transformer.py – Implements the Transformer decoder layer and the decoder; you need to implement some part of it

│

├─ grade.py – Launches some tests on the visible data and their expected output given to you

│

├─ trained_de_en_state_dict.pt – A trained Transformer encoder-decoder checkpoint for checking your implementation.

We suggest that you create a new environment with Python 3.10 for this MP to avoid conflict with other versions of PyTorch and its dependencies that you might already have installed before in your current Python environment; if you use Anaconda, you can do this by

conda create -n transformer_mp_torch_2.0.1 python=3.10

conda activate transformer_mp_torch_2.0.1

Then, in this environment,

pip install torch==2.0.1


For OSX, run

pip install torch==2.0.1 –index-url


For Linux and Windows, run

https://download.pytorch.org/whl/cpu

Then you can install the rest of the dependencies with

pip install gradescope-utils


pip install editdistance


pip install numpy


pip install jupyterlab


jupyter lab

You can now re-open this notebook by launching in the newly created

environment.

Note: for our provided visible test output to match yours, you must specify the PyTorch

torch==2.0.1

version of . Otherwise, local testing may show small discrepancies, such as

if you use another version of PyTorch. The GradeScope auto-grader will NOT show any

discrepancies as it shall be able to generate test outputs (on the hidden set) automatically.

Multi-head Attention

The multi-head attention mechanism forms the backbone of Transformer encoder and decoder layers. The paper summarizes the operation of Multi-head attention as:

4/5/24, 5:26 PM mp09_notebook

MultiHead(Q, K, V ) = Concat(head1, … , head h)W O

Q K V

where head = Attention(QWi , KWi , V Wi )

In essence, the multi-head attention module takes three inputs, the query matrix (Q), the

key matrix (K), and the value matrix (V ). Q, K, V goes through h different linear

Q

K

V

for i ∈ {1, ⋯ h}. For simplicity, we will

transformations, resulting in QWi ,

KWi

, V Wi

Tq×dmodel

Tk×dmodel

Tk

×dmodel

Q

dmodel×dk

assume that Q ∈ R

,K∈R

,V ∈R

, Wi ∈ R

,

K

dmodel×dk ,

V

dmodel×dk

and

dk

× h = dmodel

.

Wi ∈ R

Wi ∈ R

Qi

:= QWi

Ki := KWi

Vi := V Wi

For each different set of

Q,

K,

V , scaled-dot product

attention is computed as:

QiKiT



Attention(Qi, Ki, Vi) = softmax( √dk )Vi

where dk is the dimension of the individual attention heads.

Finally, Attention(Qi, Ki, Vi) from different heads i are concatenated, and the

concatenated result goes through another linear transformation W O ∈ Rdmodel×dmodel .

The figure from pg.4 of the paper shows an illustration of the operations involved.


4/5/24, 5:26 PM mp09_notebook

implement the entire operations involved in the multi-head attention mechanism. The model parameters, especially W Q, W K, W V , and W O, are defined and given in def


__init__(self, d_model, num_heads)

, and should not be modified. You should not

import other helpers not already given to you.

Note:

.

W Q, W K, and W V

are not defined as separate

nn.Linear

objects in

class

MultiHeadAttention(nn.Module)

. In

def

compute_mh_qkv_transformation(self, Q, K, V)

, you need to use

torch.Tensor.contiguous().view()

to reshape the last dimension from a single

d_model num_heads x d_k


dimension of size to two dimensions (hint: the reverse

def


operation has been defined at the last line of

compute_scaled_dot_product_attention(self, query, key, value, key_padding_mask = None, attention_mask = None)


, already given to you),

torch.Tensor.transpose()

and then use to get the expected output shape.

def compute_scaled_dot_product_attention(self, query, key,


In

value, key_padding_mask = None, attention_mask = None)

, you will also

need to correctly apply the masking operations. There are two different masks, the

key_padding_mask attention_mask


and the . Both are used to disallow attention to certain regions of the input. Please read the function definitions to figure out how to use them.


4/5/24, 5:26 PM mp09_notebook

Help on function compute_scaled_dot_product_attention in module mha:

compute_scaled_dot_product_attention(self, query, key, value, key_padding_mask =None, attention_mask=None)

This function calculates softmax(Q K^T / sqrt(d_k))V for the attention hea ds; further, a key_padding_mask is given so that padded regions are not attend ed, and an attention_mask is provided so that we can disallow attention for so me part of the sequence

Input:

query (torch.Tensor) – Query; torch tensor of size B x num_heads x T_q x d _k, where B is the batch size, T_q is the number of time steps of the query (a ka the target sequence), num_head is the number of attention heads, and d_k is the feature dimension;

key (torch.Tensor) – Key; torch tensor of size B x num_head x T_k x d_k, w here in addition, T_k is the number of time steps of the key (aka the source s equence);

value (torch.Tensor) – Value; torch tensor of size B x num_head x T_v x d_ k; where in addition, T_v is the number of time steps of the value (aka the so urce sequence);, and we assume d_v = d_k

Note: We assume T_k = T_v as the key and value come from the same source i n the Transformer implementation, in both the encoder and the decoder.

key_padding_mask (None/torch.Tensor) – If it is not None, then it is a tor ch.IntTensor/torch.LongTensor of size B x T_k, where for each key_padding_mask

for the b-th source in the batch, the non-zero positions will be ignored a s they represent the padded region during batchify operation in the dataloader (i.e., disallowed for attention) while the zero positions will be allowed for attention as they are within the length of the original sequence

attention_mask (None/torch.Tensor) – If it

is not None, then it is a torc

h.IntTensor/torch.LongTensor

of size 1 x T_q x

T_k or B x T_q x T_k, where aga

in, T_q

is the length of the

target sequence, and T_k is the length of the sou

rce sequence. An example of the attention_mask

is used for decoder self-attent

ion to enforce auto-regressive property during

parallel training; suppose the

maximum

length of a batch is

5, then the attention_mask for any input in the b

atch will

look like this for

each input of the

batch.

0

1

1

1

1

0

0

1

1

1

0

0

0

1

1

0

0

0

0

1

0

0

0

0

0

As the key_padding_mask,

the non-zero positions will be ignored and disall

owed for attention while the

zero positions will be allowed for attention.

Output:

x (torch.Tensor) – torch tensor of size B x T_q x d_model, which is the at tended output

If you believe you have implemented it correctly, you can run python.grade.py to see if you

mha.py


have passed the tests related to . We have defined four tests (out of 10, including 2

test_mha_no_mask

for EC) that you should have passed: , which tests the basic operation

mha.py test_mha_key_padding_mask


of without masking involved, , which, in addtion,

mha.py key_padding_mask


tests with ,

test_mha_key_padding_mask_attention_mask mha.py


, which, in addtion, tests

4/5/24, 5:26 PM mp09_notebook

key_padding_mask attention_mask


with and , and

test_mha_different_query_and_key mha.py


, which tests with query and key

(value) of different length in the temporal dimension.

It is expected, for now, for the other six tests to either fail or error out.


4/5/24, 5:26 PM mp09_notebook

EEE….FEE

======================================================================

ERROR: test_encoder_decoder_predictions (test_visible.TestStep.test_encoder_de

coder_predictions)

———————————————————————-

Traceback (most recent call last):

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/tests/test_visibl

e.py”, line 311, in test_encoder_decoder_predictions

output = model(src = src, tgt = trg, src_lengths = src_lengths, tgt_length

s = trg_lengths)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^^^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt

hon3.11/site-packages/torch/nn/modules/module.py”, line 1501, in _call_impl

return forward_call(*args, **kwargs)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”,

line 230, in forward

enc_output, src_padding_mask = self.forward_encoder(src, src_lengths)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”,

line 155, in forward_encoder

src_embedded = self.dropout(self.positional_encoding(self.encoder_embeddin g(src)))

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt

hon3.11/site-packages/torch/nn/modules/module.py”, line 1501, in _call_impl

return forward_call(*args, **kwargs)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt

hon3.11/site-packages/torch/nn/modules/dropout.py”, line 59, in forward

return F.dropout(input, self.p, self.training, self.inplace)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt

hon3.11/site-packages/torch/nn/functional.py”, line 1252, in dropout

return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input,

p, training)

^^^^^^^^^^^^^^^^^^

^^^^^^^^^^^^^

TypeError: dropout(): argument ‘input’ (position 1) must be Tensor, not NoneTy

pe

======================================================================

ERROR: test_encoder_decoder_states (test_visible.TestStep.test_encoder_decoder

_states)

———————————————————————-

Traceback (most recent call last):

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/tests/test_visibl e.py”, line 365, in test_encoder_decoder_states

output = model(src = src, tgt = trg, src_lengths = src_lengths, tgt_length

s = trg_lengths)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^^^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt

hon3.11/site-packages/torch/nn/modules/module.py”, line 1501, in _call_impl

return forward_call(*args, **kwargs)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”, line 230, in forward

4/5/24, 5:26 PM mp09_notebook

enc_output, src_padding_mask = self.forward_encoder(src, src_lengths) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”,

line 155, in forward_encoder

src_embedded = self.dropout(self.positional_encoding(self.encoder_embeddin

g(src)))

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt

hon3.11/site-packages/torch/nn/modules/module.py”, line 1501, in _call_impl

return forward_call(*args, **kwargs)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt

hon3.11/site-packages/torch/nn/modules/dropout.py”, line 59, in forward

return F.dropout(input, self.p, self.training, self.inplace)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt

hon3.11/site-packages/torch/nn/functional.py”, line 1252, in dropout

return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)

^^^^^^^^^^^^^^^^^^

^^^^^^^^^^^^^

TypeError: dropout(): argument ‘input’ (position 1) must be Tensor, not NoneTy

pe

======================================================================

ERROR: test_encoder_output (test_visible.TestStep.test_encoder_output)

———————————————————————-

Traceback (most recent call last):

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/tests/test_visibl

e.py”, line 272, in test_encoder_output

output_encoder, _ = model.forward_encoder(src = src, src_lengths = src_len gths)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”,

line 155, in forward_encoder

src_embedded = self.dropout(self.positional_encoding(self.encoder_embeddin

g(src)))

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt

hon3.11/site-packages/torch/nn/modules/module.py”, line 1501, in _call_impl

return forward_call(*args, **kwargs)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt

hon3.11/site-packages/torch/nn/modules/dropout.py”, line 59, in forward

return F.dropout(input, self.p, self.training, self.inplace)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt

hon3.11/site-packages/torch/nn/functional.py”, line 1252, in dropout

return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input,

p, training)

^^^^^^^^^^^^^^^^^^

^^^^^^^^^^^^^

TypeError: dropout(): argument ‘input’ (position 1) must be Tensor, not NoneTy

pe

======================================================================

ERROR: test_decoder_inference_cache_extra_credit (test_visible_ec.TestStep.tes

4/5/24, 5:26 PM mp09_notebook

t_decoder_inference_cache_extra_credit)

———————————————————————-

Traceback (most recent call last):

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/tests/test_visibl

e_ec.py”, line 162, in test_decoder_inference_cache_extra_credit

output_list, decoder_cache = model.inference(src = src, src_lengths = src_

lengths, max_output_length = MAX_INFERENCE_LENGTH)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”,

line 254, in inference

enc_output, _ = self.forward_encoder(src, src_lengths)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”,

line 155, in forward_encoder

src_embedded = self.dropout(self.positional_encoding(self.encoder_embeddin

g(src)))

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt

hon3.11/site-packages/torch/nn/modules/module.py”, line 1501, in _call_impl

return forward_call(*args, **kwargs)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt

hon3.11/site-packages/torch/nn/modules/dropout.py”, line 59, in forward

return F.dropout(input, self.p, self.training, self.inplace)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt

hon3.11/site-packages/torch/nn/functional.py”, line 1252, in dropout

return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input,

p, training)

^^^^^^^^^^^^^^^^^^

^^^^^^^^^^^^^

TypeError: dropout(): argument ‘input’ (position 1) must be Tensor, not NoneTy

pe

======================================================================

ERROR: test_decoder_inference_outputs_extra_credit (test_visible_ec.TestStep.t

est_decoder_inference_outputs_extra_credit)

———————————————————————-

Traceback (most recent call last):

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/tests/test_visibl

e_ec.py”, line 109, in test_decoder_inference_outputs_extra_credit

output_list, decoder_cache = model.inference(src = src, src_lengths = src_

lengths, max_output_length = MAX_INFERENCE_LENGTH)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”,

line 254, in inference

enc_output, _ = self.forward_encoder(src, src_lengths)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”,

line 155, in forward_encoder

src_embedded = self.dropout(self.positional_encoding(self.encoder_embeddin

g(src)))

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt

hon3.11/site-packages/torch/nn/modules/module.py”, line 1501, in _call_impl

return forward_call(*args, **kwargs)

4/5/24, 5:26 PM mp09_notebook

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt hon3.11/site-packages/torch/nn/modules/dropout.py”, line 59, in forward

return F.dropout(input, self.p, self.training, self.inplace) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt hon3.11/site-packages/torch/nn/functional.py”, line 1252, in dropout

return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)

^^^^^^^^^^^^^^^^^^

^^^^^^^^^^^^^

TypeError: dropout(): argument ‘input’ (position 1) must be Tensor, not NoneTy pe

====================================================================== FAIL: test_pe (test_visible.TestStep.test_pe)

———————————————————————-

Traceback (most recent call last):

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/tests/test_visibl e.py”, line 236, in test_pe

self.assertAlmostEqual(torch.sum(torch.abs(pe.pe – self.pe_test_data[“p e”])).item(), 0, places = 4, msg=’Positional Encoding has incorrect encoding e ntries’)

AssertionError: 270.70733642578125 != 0 within 4 places (270.70733642578125 di

fference) : Positional Encoding has incorrect encoding entries

———————————————————————-

Ran 10 tests in 1.604s

FAILED (failures=1, errors=5)

Positional Encoding

In order for the model to make use of the order of the sequence, some information about

the relative or absolute position of the tokens in the sequence must be injected into the

input embeddings at the bottoms of the encoder and decoder stacks. These positional

encodings have the same dimension dmodel as the embeddings and the rest of the encoder

and decoder modules. The original paper defines a simple positional encoding as sine and cosine functions of different frequencies:

PE(pos,2i) = sin(pos /100002i/dmodel )

PE(pos,2i+1) = cos(pos /100002i/dmodel )

where pos is the position and i is an index into the dimension dmodel (i.e.,

∈ {0, 1, ⋯ , dmodel}). That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 × 2π.

pe.py


In , you will need to fill in the missing code to calculate the positional encoding,

self.pe def __init__(self, d_model, max_seq_length)


, in and implement the

def forward(x) class PositionalEncoding(nn.Module)


function in the class .

Note:

4/5/24, 5:26 PM mp09_notebook

For better numerical accuracy, it is recommended that you make use of:

4/5/24, 5:26 PM mp09_notebook

EEF…..EE

======================================================================

ERROR: test_encoder_decoder_predictions (test_visible.TestStep.test_encoder_de coder_predictions)

———————————————————————-

Traceback (most recent call last):

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/tests/test_visibl e.py”, line 311, in test_encoder_decoder_predictions

output = model(src = src, tgt = trg, src_lengths = src_lengths, tgt_length s = trg_lengths)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt hon3.11/site-packages/torch/nn/modules/module.py”, line 1501, in _call_impl

return forward_call(*args, **kwargs)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”, line 233, in forward

dec_output = self.forward_decoder(enc_output, src_padding_mask, tgt, tgt_l engths)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”, line 208, in forward_decoder

return dec_output

^^^^^^^^^^

NameError: name ‘dec_output’ is not defined

======================================================================

ERROR: test_encoder_decoder_states (test_visible.TestStep.test_encoder_decoder _states)

———————————————————————-

Traceback (most recent call last):

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/tests/test_visibl e.py”, line 365, in test_encoder_decoder_states

output = model(src = src, tgt = trg, src_lengths = src_lengths, tgt_length s = trg_lengths)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt hon3.11/site-packages/torch/nn/modules/module.py”, line 1501, in _call_impl

return forward_call(*args, **kwargs)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”, line 233, in forward

dec_output = self.forward_decoder(enc_output, src_padding_mask, tgt, tgt_l engths)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”, line 208, in forward_decoder

return dec_output

^^^^^^^^^^

NameError: name ‘dec_output’ is not defined

======================================================================

ERROR: test_decoder_inference_cache_extra_credit (test_visible_ec.TestStep.tes t_decoder_inference_cache_extra_credit)

———————————————————————-

Traceback (most recent call last):

4/5/24, 5:26 PM mp09_notebook

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/tests/test_visibl e_ec.py”, line 162, in test_decoder_inference_cache_extra_credit

output_list, decoder_cache = model.inference(src = src, src_lengths = src_

lengths, max_output_length = MAX_INFERENCE_LENGTH)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”,

line 274, in inference

tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embeddin

g(tgt)))

^^^

NameError: name ‘tgt’ is not defined

======================================================================

ERROR: test_decoder_inference_outputs_extra_credit (test_visible_ec.TestStep.t

est_decoder_inference_outputs_extra_credit)

———————————————————————-

Traceback (most recent call last):

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/tests/test_visibl

e_ec.py”, line 109, in test_decoder_inference_outputs_extra_credit

output_list, decoder_cache = model.inference(src = src, src_lengths = src_

lengths, max_output_length = MAX_INFERENCE_LENGTH)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”,

line 274, in inference

tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embeddin

g(tgt)))

^^^

NameError: name ‘tgt’ is not defined

====================================================================== FAIL: test_encoder_output (test_visible.TestStep.test_encoder_output)

———————————————————————-

Traceback (most recent call last):

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/tests/test_visibl

e.py”, line 279, in test_encoder_output

self.assertAlmostEqual(torch.sum(torch.abs(ref_enc_item – enc_item)).item

(), 0, places = 4, msg=f’The encoder output for the sample #{item_idx} in the

batch #{it_idx} is not correct’)

AssertionError: 646.3253784179688 != 0 within 4 places (646.3253784179688 diff

erence) : The encoder output for the sample #0 in the batch #0 is not correct

———————————————————————-

Ran 10 tests in 1.817s

FAILED (failures=1, errors=4)

Transformer Encoder

The Transformer Encoder consists of a stack of Transformer encoder layers. Each encoder

layer has two sets of sub-layer operations, applied one set after another. The first set

involves the multi-head self-attention mechanism, and the second set involves a simple,

position-wise, fully connected feed-forward network. In the paper, for each of the two sub-

layer operation sets, a dropout operation is immediately applied to the output of the sub-

4/5/24, 5:26 PM mp09_notebook


Out[14]:

4/5/24, 5:26 PM mp09_notebook


encoder.py def forward(self,


In , you will need to complete the

x,self_attn_padding_mask = None) class


defined in the class

TransformerEncoderLayer(nn.Module)

, which implements a single layer in the

def __init__(self, embedding_dim,


Transformer encoder. In

ffn_embedding_dim, num_attention_heads, dropout_prob)

, we have already pre-

defined some model submodules and hyperparameters as:

self.embedding_dim – This is the model dimension, aka d_model

self.self_attn – Build the self-attention mechanism using MultiHeadAttention implemented earlier

self.self_attn_layer_norm – Layer norm for the self-attention layer’s output

self.activation_fn – The ReLU activation for position-wise feed-forward network

self.fc1 – The parameters will be used for W_1 and b_1 in the position-wise feed-forward network

self.fc2 – The parameters will be used for W_2 and b_2 in the position-wise feed-forward network

self.dropout – The DropOut regularization module to be applied immediately after self-attention module and FFN module

def forward(self, x,self_attn_padding_mask = None)


As described earlier, in , you are simply asked to implement:

LayerNorm(x + DropOut(Self-Attention(x)))

followed by

LayerNorm(x + DropOut(FFN(x)))

where FFN(x) = max (0, xW1 + b1) W2 + b2.

class


You should use all the model parameters already given to you in

TransformerEncoderLayer(nn.Module)

, and should not need to define or use other

parameters or helper functions.

The entire Transformer encoder has already been implemented for you, which simply stacks the Transformer Encoder Layer implemented earlier together to form a Transformer

4/5/24, 5:26 PM mp09_notebook

4/5/24, 5:26 PM mp09_notebook

some German sentences (that are converted to discrete index sequences) to check against the intermediate encoder output pre-generated by our implementation.

It is expected, for now, for the other four tests to either fail or error out (assuming you

mha.py pe.py


passed the four tests for , and the one test for already)

Note: if you are getting a very small error that causes you to fail the test, please double-check that you have the correct torch version installed. Our outputs on the visible set are

Python 3.10 torch== 2.0.1

pre-generated using and , so at the very least you need

torch==2.0.1

to make sure that you are using for this. It has been known that newer or older PyTorch versions may have slightly different implementations of the same internal module, and result in slightly different computed results. If in doubt, you can also submit to GradeScope for testing. The auto-grader on GradeScope will NOT have this issue, as the solutions will be generated during the submission with the same platform and package versions as your implementation will use, but your code will be tested on the hidden set instead.


4/5/24, 5:26 PM mp09_notebook

EE……EE

======================================================================

ERROR: test_encoder_decoder_predictions (test_visible.TestStep.test_encoder_de coder_predictions)

———————————————————————-

Traceback (most recent call last):

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/tests/test_visibl e.py”, line 311, in test_encoder_decoder_predictions

output = model(src = src, tgt = trg, src_lengths = src_lengths, tgt_length s = trg_lengths)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt hon3.11/site-packages/torch/nn/modules/module.py”, line 1501, in _call_impl

return forward_call(*args, **kwargs)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”, line 233, in forward

dec_output = self.forward_decoder(enc_output, src_padding_mask, tgt, tgt_l engths)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”, line 208, in forward_decoder

return dec_output

^^^^^^^^^^

NameError: name ‘dec_output’ is not defined

======================================================================

ERROR: test_encoder_decoder_states (test_visible.TestStep.test_encoder_decoder _states)

———————————————————————-

Traceback (most recent call last):

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/tests/test_visibl e.py”, line 365, in test_encoder_decoder_states

output = model(src = src, tgt = trg, src_lengths = src_lengths, tgt_length s = trg_lengths)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^

File “/home/jerome-ni/anaconda3/envs/test_transformer_mp_torch_2.0.1/lib/pyt hon3.11/site-packages/torch/nn/modules/module.py”, line 1501, in _call_impl

return forward_call(*args, **kwargs)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”, line 233, in forward

dec_output = self.forward_decoder(enc_output, src_padding_mask, tgt, tgt_l engths)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”, line 208, in forward_decoder

return dec_output

^^^^^^^^^^

NameError: name ‘dec_output’ is not defined

======================================================================

ERROR: test_decoder_inference_cache_extra_credit (test_visible_ec.TestStep.tes t_decoder_inference_cache_extra_credit)

———————————————————————-

Traceback (most recent call last):

4/5/24, 5:26 PM mp09_notebook


Out[15]:

decoder.py def forward(self, x, encoder_out


In , you will need to complete the

= None, encoder_padding_mask = None, self_attn_padding_mask =


None,self_attn_mask = None) class


defined in the class

TransformerDecoderLayer(nn.Module)

, which implements a single decoder layer in

def __init__(self, embedding_dim,


the Transformer decoder. In

ffn_embedding_dim, num_attention_heads, dropout_prob, no_encoder_attn=False)


we have already pre-defined some model submodules and

hyperparameters as:

4/5/24, 5:26 PM mp09_notebook

self.embedding_dim – This is the model dimension, aka d_model

self.self_attn – Build the decoder self-attention mechanism using MultiHeadAttention implemented earlier

self.self_attn_layer_norm – Layer norm for the decoder self-attention layer’s output

self.encoder_attn – If an encoder-decoder architecture is built, we build the encoder-decoder attention using MultiHeadAttention implemented earlier

self.encoder_attn_layer_norm – Layer norm for the encoder-decoder attention layer’s output

self.activation_fn – The ReLU activation for position-wise feed-forward network

self.fc1 – The parameters will be used for W_1 and b_1 in the position-wise feed-forward network

self.fc2 – The parameters will be used for W_2 and b_2 in the position-wise feed-forward network

self.dropout – The DropOut regularization module to be applied immediately after self-attention module, encoder-decoder attention module and FFN module

def forward(self, x, encoder_out = None,


As described earlier, in

encoder_padding_mask = None, self_attn_padding_mask = None,self_attn_mask = None)


you are simply asked to implement:

x = LayerNorm(x + DropOut(Decoder-Self-Attention(x)))

followed by

= LayerNorm(x + DropOut(Encoder-Decoder-Attention(x, encoder_out)))

followed by

= LayerNorm(x + DropOut(FFN(x)))

where FFN(x) = max (0, xW1 + b1) W2 + b2.

class


You should use all the model parameters already given to you in

TransformerDecoderLayer(nn.Module)

, and should not need to define or use other

parameters or helper functions.

The entire Transformer decoder has already been implemented for you, which simply stacks the TransformerDecoderLayer implemented earlier together to form a TransformerDecoder.

4/5/24, 5:26 PM mp09_notebook

4/5/24, 5:26 PM mp09_notebook

Help on function forward in module decoder:

forward(self, x, encoder_out=None, encoder_padding_mask=None, self_attn_paddin g_mask=None, self_attn_mask=None)

Applies the self attention module + Dropout + Add & Norm operation, the en coder-decoder attention + Dropout + Add & Norm operation (if self.encoder_attn is not None), and the position-wise feedforward network + Dropout + Add & Norm operation. Note that LayerNorm is applied after the self-attention operation, after the encoder-decoder attention operation and another time after the ffn m odules, similar to the original Transformer implementation.

Input:

x (torch.Tensor) – input tensor of size B x T_d x embedding_dim from t he decoder input or the previous encoder layer, where T_d is the decoder’s tem poral dimension; serves as input to the TransformerDecoderLayer’s self attenti on mechanism.

encoder_out (None/torch.Tensor) – If it is not None, then it is the ou tput from the TransformerEncoder as a tensor of size B x T_e x embedding_dim, where T_e is the encoder’s temporal dimension; serves as part of the input to the TransformerDecoderLayer’s self attention mechanism (hint: which part?).

encoder_padding_mask (None/torch.Tensor) – If it is not None, then it is a torch.IntTensor/torch.LongTensor of size B x T_e, where for each encoder_ padding_mask[b] for the b-th source in the batched tensor encoder_out[b], the non-zero positions will be ignored as they represent the padded region during batchify operation in the dataloader (i.e., disallowed for attention) while th e zero positions will be allowed for attention as they are within the length o f the original sequence

self_attn_padding_mask (None/torch.Tensor) – If it is not None, then i t is a torch.IntTensor/torch.LongTensor of size B x T_d, where for each self_a ttn_padding_mask[b] for the b-th source in the batched tensor x[b], the non-ze ro positions will be ignored as they represent the padded region during batchi fy operation in the dataloader (i.e., disallowed for attention) while the zero positions will be allowed for attention as they are within the length of the o riginal sequence

self_attn_mask (None/torch.Tensor) – If it is not None, then it is a t orch.IntTensor/torch.LongTensor of size 1 x T_d x T_d or B x T_d x T_d. It is used for decoder self-attention to enforce auto-regressive property during par allel training; suppose the maximum length of a batch is 5, then the attention _mask for any input in the batch will look like this for each input of the bat ch.

01111

00111

00011

00001

00000

The non-zero positions will be ignored and disallowed for attention wh ile the zero positions will be allowed for attention.

Output:

x (torch.Tensor) – the decoder layer’s output, of size B x T_d x embed ding_dim, after the self attention module + Dropout + Add & Norm operation, th e encoder-decoder attention + Dropout + Add & Norm operation (if self.encoder_ attn is not None), and the position-wise feedforward network + Dropout + Add & Norm operation.

4/5/24, 5:26 PM mp09_notebook

class


To finish your Transformer decoder (after reading the implementation in

TransformerDecoder(nn.Module) def


), you now need to implement the

forward_decoder(self, enc_output, src_padding_mask, tgt, tgt_lengths)

class Transformer(nn.Module) transformer.py

in , which is defined in . The

__init__(self, src_vocab_size, tgt_vocab_size, sos_idx, eos_idx,


d_model, num_heads, num_layers, d_ff, max_seq_length, dropout_prob)

defines some model submodules and hyperparameters that are related to

forward_decoder

as:

self.decoder_embedding – Decoder input embedding that converts discrete tokens into continuous vectors; this will be already invoked for you in forward_decoder to get the decoder’s input tgt_embedded.

self.positional_encoding – Positional Encoding used by the Transformer decoder; this will be already invoked for you in forward_decoder to get the decoder’s input tgt_embedded.

self.decoder – Creates an instance of TransformerDecoder.

self.dropout – For applying additional dropout after positional encoding; this will be already invoked for you in forward_decoder to get the decoder’s input tgt_embedded.

self.sos_idx – Every encoder and decoder sequence starts with this token index (useful in extra-credit)

self.eos_idx – Every encoder and decoder sequence ends with this token index (useful in extra-credit)

4/5/24, 5:26 PM mp09_notebook

implementation, assuming ALL your previous tests have passed. We have defined two tests

test_encoder_decoder_predictions

that you should have passed: , which initializes a

Transformer

object, loads the model weights from a de-en neural machine translation

trained_de_en_state_dict.pt forward_decoder

checkpoint , and invokes the

function on some German sentences (that are converted to discrete index sequences),

which, conditioned on previous tokens in the parallel English sentences, predicts the next tokens in English. We take your next token prediction outputs of the unpadded regions to check against the next token predictions pre-generated by our implementation. This test

will also throw an error if the next token prediction accuracy does not match ours, which

test_encoder_decoder_states,

indicates some major bugs. The second test, works similarly, except that we are checking your pre-softmax output layer states from the decoder instead of discrete predictions.

It is expected, for now, for the other two tests to either fail or error out, as they are for extra

mha.py pe.py


credit (assuming you passed the four tests for , the one test for , and the one test for your Transformer encoder implementation).

Note: if you are getting a very small error that causes you to fail the test, please double-

check that you have the correct torch version installed. Our outputs on the visible set are

Python 3.10 torch== 2.0.1

pre-generated using and , so at the very least you need

torch==2.0.1

to make sure that you are using for this. It has been known that newer or older PyTorch versions may have slightly different implementations of the same internal module, and result in slightly different computed results. If in doubt, you can also submit to GradeScope for testing. The auto-grader on GradeScope will NOT have this issue, as the solutions will be generated during the submission with the same platform and package versions as your implementation will use, but your code will be tested on the hidden set instead.


4/5/24, 5:26 PM mp09_notebook

……..EE

======================================================================

ERROR: test_decoder_inference_cache_extra_credit (test_visible_ec.TestStep.tes

t_decoder_inference_cache_extra_credit)

———————————————————————-

Traceback (most recent call last):

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/tests/test_visibl

e_ec.py”, line 162, in test_decoder_inference_cache_extra_credit

output_list, decoder_cache = model.inference(src = src, src_lengths = src_

lengths, max_output_length = MAX_INFERENCE_LENGTH)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”,

line 274, in inference

tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embeddin g(tgt)))

^^^

NameError: name ‘tgt’ is not defined

======================================================================

ERROR: test_decoder_inference_outputs_extra_credit (test_visible_ec.TestStep.t

est_decoder_inference_outputs_extra_credit)

———————————————————————-

Traceback (most recent call last):

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/tests/test_visibl

e_ec.py”, line 109, in test_decoder_inference_outputs_extra_credit

output_list, decoder_cache = model.inference(src = src, src_lengths = src_

lengths, max_output_length = MAX_INFERENCE_LENGTH)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File “/mnt/c/Users/junru/Downloads/transformer_mp/src_test/transformer.py”,

line 274, in inference

tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embeddin

g(tgt)))

^^^

NameError: name ‘tgt’ is not defined

———————————————————————-

Ran 10 tests in 2.064s

FAILED (errors=2)

Submission

For submission onto GradeScope, make sure you are uploading all relevant Python modules

onto Gradescope. Again, these are:

mha.py

pe.py

encoder.py

decoder.py

transformer.py

4/5/24, 5:26 PM mp09_notebook

Extra Credit: Auto-regressive Decoding During Inference

forward

In the main part of the MP, we implemented the function of the Transformer

decoder. The previous implementation is mainly used in training. During training, when we

are given paired source and target sequences, we feed the source sequence into the

encoder, we feed the entire target sequence as the decoder’s input, and we predict the left-

shifted target sequence with the decoder auto-regressive attention mask applied

(conditioned on the encoder’s output). Hence, we call this training routine “teacher-forcing”.

This is depicted in the following figure, where we use [x1, x2, x3, x4] to indicate the source sequence and [y1, y2, y3] to indicate the paired target sequence, as one would find in most

sequence-to-sequence parallel datasets. We introduce the < sos > token as the start-of-sentence sentinel and the < eos > token as the end-of-sentence sentinel. Note that with

our decoder auto-regressive attention mask implemented earlier (and passed to the decoder during training), when predicting y1, besides the entire encoder states

[¯x1, x¯2, x¯3, x¯4], it can only access the decoder states of < sos > for decoder self-

attention; when predicting y2, besides the entire encoder states [x¯1, x¯2, x¯3, x¯4], it can only

access the decoder states of < sos > and y1 for decoder self-attention; when predicting y3, besides the entire encoder states [¯x1, x¯2, x¯3, x¯4], it can only access the decoder states of < sos >, y1 and y2 for decoder self-attention; when predicting < eos >, besides the entire encoder states [¯x1, x¯2, x¯3, x¯4], it can access the decoder states of < sos > , y1, y2 and y3 for decoder self-attention.

Note that even if the decoder were to make a mistake after the output layer during training,

~

and suppose that it predicts y2 as y2, when predicting y3, the decoder prediction is still

conditioned on the decoder states generated by < sos >, y1 and y2, instead of < sos >,

~

y1 and y2. This is where the name “teacher-forcing” came from.


4/5/24, 5:26 PM mp09_notebook

Encoder Input Sequence Decoder Input Sequence

To complete the extra credit, you will need to complete the missing code in three functions.

def forward_one_step_ec(self, x, encoder_out = None,


The first function is

encoder_padding_mask = None, self_attn_padding_mask = None, self_attn_mask = None, cache = None) class


in

TransformerDecoderLayer(nn.Module) decoder.py

, within . Please read the

function helper and the comments within.


4/5/24, 5:26 PM mp09_notebook

Help on function forward_one_step_ec in module decoder:

forward_one_step_ec(self, x, encoder_out=None, encoder_padding_mask=None, self _attn_padding_mask=None, self_attn_mask=None, cache=None)

Applies the self attention module + Dropout + Add & Norm operation, the en coder-decoder attention + Dropout + Add & Norm operation (if self.encoder_attn is not None), and the position-wise feedforward network + Dropout + Add & Norm operation, but for just a single time step at the last time step. Note that La yerNorm is applied after the self-attention operation, after the encoder-decod er attention operation and another time after the ffn modules, similar to the original Transformer implementation.

Input:

x (torch.Tensor) – input tensor of size B x T_d x embedding_dim from t he decoder input or the previous encoder layer, where T_d is the decoder’s tem poral dimension; serves as input to the TransformerDecoderLayer’s self attenti on mechanism. You need to correctly slice x in the function below so that it i s only calculating a one-step (one frame in length in the temporal dimension) decoder output of the last time step.

encoder_out (None/torch.Tensor) – If it is not None, then it is the ou tput from the TransformerEncoder as a tensor of size B x T_e x embedding_dim, where T_e is the encoder’s temporal dimension; serves as part of the input to the TransformerDecoderLayer’s self attention mechanism (hint: which part?).

encoder_padding_mask (None/torch.Tensor) – If it is not None, then it is a torch.IntTensor/torch.LongTensor of size B x T_e, where for each encoder_ padding_mask[b] for the b-th source in the batched tensor encoder_out[b], the non-zero positions will be ignored as they represent the padded region during batchify operation in the dataloader (i.e., disallowed for attention) while th e zero positions will be allowed for attention as they are within the length o f the original sequence

self_attn_padding_mask (None/torch.Tensor) – If it is not None, then i t is a torch.IntTensor/torch.LongTensor of size B x T_d, where for each self_a ttn_padding_mask[b] for the b-th source in the batched tensor x[b], the non-ze ro positions will be ignored as they represent the padded region during batchi fy operation in the dataloader (i.e., disallowed for attention) while the zero positions will be allowed for attention as they are within the length of the o riginal sequence. If it is not None, then you need to correctly slice it in th e function below so that it is corresponds to the self_attn_padding_mask for c alculating a one-step (one frame in length in the temporal dimension) decoder output of the last time step.

self_attn_mask (None/torch.Tensor) – If it is not None, then it is a t orch.IntTensor/torch.LongTensor of size 1 x T_d x T_d or B x T_d x T_d. It is used for decoder self-attention to enforce auto-regressive property during par allel training; suppose the maximum length of a batch is 5, then the attention _mask for any input in the batch will look like this for each input of the bat ch.

01111

00111

00011

00001

00000

The non-zero positions will be ignored and disallowed for attention wh ile the zero positions will be allowed for attention. If it is not None, then you need to correctly slice it in the function below so that it is corresponds to the self_attn_padding_mask for calculating a one-step (one frame in length in the temporal dimension) decoder output of the last time step.

4/5/24, 5:26 PM mp09_notebook

cache (torch.Tensor) – the output from this decoder layer previously c omputed up until the previous time step before the last; hence it is of size B x (T_d-1) x embedding_dim. It is to be concatenated with the single time-step output calculated in this function before being returned

Returns:

x (torch.Tensor) – Output tensor B x T_d x embedding_dim, which is a c oncatenation of cache (previously computed up until the previous time step bef ore the last) and the newly computed one-step decoder output for the last time step.

forward(self, x, encoder_out=None,


It is very similar to the

encoder_padding_mask=None, self_attn_padding_mask=None,


self_attn_mask=None) class TransformerDecoderLayer(nn.Module)

in , except

for a few things:

x


Whileis still an input tensor of size (B, T_d, embedding_dim), when calculating the

x


decoder self-attention, you need to slice the input and the input masks so that the

ag_x


last frame of shape (B , 1 ,embedding_dim) shall attend to all temporal

x


dimensions of , and the output of decoder self-attention, encoder-decoder attention, and position-wise feedforward network are all of the shape (B , 1 ,embedding_dim).

cache

. It has an extra input argument called . This is the decoder layer output for the current decoder layer computed until and including the previous time step. Hence, the temporal dimension is one less than the temporal dimension of the decoder input

x


argument . Once you have computed the layer output of the current time step, you need to concatenate your computed one-frame-length output of size (B , 1

cache

,embedding_dim) to the temporal dimension of , and return it as the return value.

def forward_one_step_ec(self, x,


The second function you need to write is

decoder_padding_mask = None, decoder_attention_mask = None,


encoder_out = None, encoder_padding_mask = None, cache = None) class


in

TransformerDecoder(nn.Module) decoder.py

, within . Please read the function

helper and the comments within.


4/5/24, 5:26 PM mp09_notebook

Help on function forward_one_step_ec in module decoder:

forward_one_step_ec(self, x, decoder_padding_mask=None, decoder_attention_mask =None, encoder_out=None, encoder_padding_mask=None, cache=None)

Forward one step.

Input:

x (torch.Tensor) – input tensor of size B x T_d x embedding_dim; input to the TransformerDecoderLayer’s self attention mechanism

decoder_padding_mask (None/torch.Tensor) – If it is not None, then it is a torch.IntTensor/torch.LongTensor of size B x T_d, where for each decoder_ padding_mask[b] for the b-th source in the batched tensor x[b], the non-zero p ositions will be ignored as they represent the padded region during batchify o peration in the dataloader (i.e., disallowed for attention) while the zero pos itions will be allowed for attention as they are within the length of the orig inal sequence

decoder_attention_mask (None/torch.Tensor) – If it is not None, then i t is a torch.IntTensor/torch.LongTensor of size 1 x T_d x T_d or B x T_d x T_ d. It is used for decoder self-attention to enforce auto-regressive property d uring parallel training; suppose the maximum length of a batch is 5, then the attention_mask for any input in the batch will look like this for each input o f the batch.

01111

00111

00011

00001

00000

The non-zero positions will be ignored and disallowed for attention wh ile the zero positions will be allowed for attention.

encoder_out (None/torch.Tensor) – If it is not None, then it is the ou tput from the TransformerEncoder as a tensor of size B x T_e x embedding_dim, where T_e is the encoder’s temporal dimension; serves as part of the input to the TransformerDecoderLayer’s self attention mechanism (hint: which part?).

encoder_padding_mask (None/torch.Tensor) – If it is not None, then it is a torch.IntTensor/torch.LongTensor of size B x T_e, where for each encoder_ padding_mask[b] for the b-th source in the batch, the non-zero positions will be ignored as they represent the padded region during batchify operation in th e dataloader (i.e., disallowed for attention) while the zero positions will be allowed for attention as they are within the length of the original sequence

cache (None/List[torch.Tensor]) – If it is not None, then it is a lis t of cache tensors of each decoder layer calculated until and including the pr evious time step; hence, if it is not None, then each tensor in the list is of size B x (T_d-1) x embedding_dim; the list length is equal to len(self.layer s), or the number of decoder layers.

Output:

y (torch.Tensor) – Output tensor from the Transformer decoder consist ing of a single time step, of size B x 1 x embedding_dim, if output layer is N one, or of size B x 1 x output_layer_size, if there is an output layer.

new_cache (List[torch.Tensor]) – List of cache tensors of each decode r layer for use by the auto-regressive decoding of the next time step; each te nsor is of size B x T_d x embedding_dim; the list length is equal to len(self. layers), or the number of decoder layers.

4/5/24, 5:26 PM mp09_notebook

Help on function inference in module transformer:

inference(self, src, src_lengths, max_output_length)

Applies the entire Transformer encoder-decoder to src and target, possibly

as used during inference to auto-regressively obtain the next token; each sequ

ence in src has been padded to the max(src_lengths)

Input:

src (torch.Tensor) – Encoder’s input tensor of size B x T_e x d_model

src_lengths (torch.Tensor) – A 1D iterable of Long/Int of length B, wh ere the b-th length in src_lengths corresponds to the actual length of src[b] (beyond that is the pre-padded region); T_e = max(src_lengths)

Output:

decoded_list (List(torch.Tensor) – a list of auto-regressively obtaine d decoder output token predictions; the b-th item of the decoded_list should b e the output from src[b], and each of the sequence predictions in decoded_list is of a possibly different length.

decoder_layer_cache_list (List(List(torch.Tensor))) – a list of decode r_layer_cache; the b-th item of the decoded_layer_cache_list should be the dec oder_layer_cache for the src[b], which itself is a list of torch.Tensor, as re turned by self.decoder.forward_one_step_ec (see the function definition there for more details) when the auto-regressive inference ends for src[b].

If you believe you have implemented everything in this section correctly, you can run

python.grade.py

to see if you have passed the test related to the extra credit auto-

regressive inference implementation, assuming ALL your previous tests have passed. We

have defined two tests that you should have passed:

test_decoder_inference_outputs_extra_credit

, which initializes a

Transformer

object, loads the model weights from a de-en neural machine translation

trained_de_en_state_dict.pt inference

checkpoint , and invokes the function on some German sentences (that are converted to discrete index sequences), which auto-

regressively decodes the English translation (that are in the form of discrete index

decoded_list inference

sequences). We take your returned by to check against our decoded results pre-generated by our implementation. This test will also throw an error if the normalized edit distance, or error rate computed against ground-truth English

translations, does not match ours, which indicates some major bugs. The second test,

test_decoder_inference_cache_extra_credit,

works similarly, except that we are

decoder_layer_cache_list inference

checking the returned by function, which is a more careful examination of whether your auto-regressive inference has been

implemented correctly.

Note: if you are getting a very small error that causes you to fail the test, please double-check that you have the correct torch version installed. Our outputs on the visible set are

Python 3.10 torch== 2.0.1

pre-generated using and , so at the very least you need

torch==2.0.1

to make sure that you are using for this. It has been known that newer or older PyTorch versions may have slightly different implementations of the same internal module, and result in slightly different computed results. If in doubt, you can also submit to

4/5/24, 5:26 PM mp09_notebook

GradeScope for testing. The auto-grader on GradeScope will NOT have this issue, as the solutions will be generated during the submission with the same platform and package versions as your implementation will use, but your code will be tested on the hidden set instead.


https://courses.grainger.illinois.edu/ece448 /mp/mp09_notebook.html 42/42
