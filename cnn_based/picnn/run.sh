# To check the result from the pretrained model in log
#python net.py --dataset 'ModelNet' --gpu_num 1 --alpha 1.0 --beta 1.0 --trial 1 -e 40 --train_f
#python net.py --dataset 'ObjectPI' --gpu_num 1 --alpha 1.0 --beta 1.0 --trial 1 -e 40 --train_f

#To check robustness to the number of views in log_robustnesss
python net_robustness.py --dataset 'ModelNet' --trial 1 --alpha 1.0  --beta 1.0 --batchSize 8 --gpu_num 1
python net_robustness.py --dataset 'ObjectPI'  --trial 1 --alpha 1.0  --beta 1.0 --batchSize 8 --gpu_num 1

# To train your own model
# python net.py --dataset 'ModelNet' --gpu_num 1 --alpha 1.0 --beta 1.0 --trial 1 -e 40
# python net.py --dataset 'ObjectPI' --gpu_num 1 --alpha 1.0 --beta 1.0 --trial 1 -e 40