python train.py --cuda --mb 1024 --epoch 200 --n_negs 5 --name norm_neg5 > ./log/norm_neg5.txt
python predict.py --sememe --name norm_neg5 >> ./log/norm_neg5.txt
python eval.py --result norm_neg5_result.txt >> ./log/norm_neg5.txt