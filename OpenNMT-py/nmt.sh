python preprocess.py -train_src ../../Data/iwslt+wmt/train.bpe.en -train_tgt ../../Data/iwslt+wmt/train.bpe.zh -valid_src ../../Data/iwslt+wmt/dev.bpe.en -valid_tgt ../../Data/iwslt+wmt/dev.bpe.zh -save_data ../../Exp/Data/iwslt_wmt/bpe -tgt_seq_length 80 -src_vocab ../../Data/iwslt+wmt/vocab.bpe.en
python  train.py -data ../../Exp/Data/iwslt_wmt/bpe -save_model ../../Exp/Model/iwslt_wmt/bpe -layers 6 -rnn_size 512 -word_vec_size 512  -transformer_ff 2048 -heads 8 -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 200000000  -max_generator_batches 2 -dropout 0.1 -batch_size 2048 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -learning_rate 2 -decay_method noam -warmup_steps 500000 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000  -world_size 2 -gpu_ranks 0 1 -tensorboard -tensorboard_log_dir ../../Exp/Log/WMT -valid_batch_size 1 -decay_steps 1000000