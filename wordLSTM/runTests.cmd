python main.py --data ./data/shortjokes_full/ --model LSTM --emsize 1000 --vocsize 62000 --nhid 256 --nlayers 2 --epochs 6 --batch_size 20 --bptt 20 --cuda --save voc62k_batch20_em1000_lay2_hid256_ep6_bptt20.mod

python main.py --data ./data/shortjokes_full/ --model LSTM --emsize 300 --vocsize 62000 --nhid 256 --nlayers 2 --epochs 6 --batch_size 20 --bptt 20 --cuda --save voc62k_batch20_em300_lay2_hid256_ep6_bptt20.mod

python main.py --data ./data/shortjokes_full/ --model LSTM --emsize 500 --vocsize 62000 --nhid 256 --nlayers 2 --epochs 6 --batch_size 20 --bptt 20 --cuda --save voc62k_batch20_em500_lay2_hid256_ep6_bptt20.mod

python main.py --data ./data/shortjokes_full/ --model LSTM --emsize 700 --vocsize 62000 --nhid 256 --nlayers 2 --epochs 6 --batch_size 20 --bptt 20 --cuda --save voc62k_batch20_em700_lay2_hid256_ep6_bptt20.mod


python main.py --data ./data/shortjokes_full/ --model LSTM --emsize 700 --vocsize 62000 --nhid 256 --nlayers 2 --epochs 6 --batch_size 15 --bptt 35 --cuda --save voc62k_batch15_em700_lay2_hid256_ep6_bptt35.mod

python main.py --data ./data/shortjokes_full/ --model LSTM --emsize 500 --vocsize 62000 --nhid 500 --nlayers 2 --epochs 6 --batch_size 10 --bptt 20 --cuda --save voc62k_batch10_em500_lay2_hid500_ep6_bptt20.mod

python main.py --data ./data/shortjokes_full/ --model LSTM --emsize 500 --vocsize 62000 --nhid 500 --nlayers 2 --epochs 6 --batch_size 20 --bptt 35 --cuda --save voc62k_batch20_em500_lay2_hid500_ep6_bptt35.mod
