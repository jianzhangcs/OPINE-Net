python Train_CS_OPINE_Net_plus.py --cs_ratio 25 --layer_num 9 --learning_rate 1e-4 --start_epoch 0   --end_epoch 160 --gpu_list 0
python Train_CS_OPINE_Net_plus.py --cs_ratio 25 --layer_num 9 --learning_rate 1e-5 --start_epoch 160 --end_epoch 170 --gpu_list 0 --save_interval 1

python TEST_CS_OPINE_Net_plus.py  --cs_ratio 25 --layer_num 9 --epoch_num 170

