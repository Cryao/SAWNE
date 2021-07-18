# SAWNE
## How to Run

Using the following command to train/predict/evaluate.

```sh
python train.py --cuda --mb 1024 --epoch 200 --n_negs 20 --name norm_neg20 > ./log/norm_neg20.txt
python predict.py --sememe --name norm_neg20 >> ./log/norm_neg20.txt
python eval.py --result norm_neg20_result.txt >> ./log/norm_neg20.txt
```