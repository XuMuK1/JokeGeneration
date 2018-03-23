python generate.py --data ./data/shortjokes_full/ --dictionary currentDict.pkl --cuda --checkpoint voc62k_batch20_em1000_lay2_hid256_ep6_bptt20.mod --temperature 0.5 --outf ./outs/voc62k_batch20_em1000_lay2_hid256_ep6_bptt20_gen05.txt
python generate.py --data ./data/shortjokes_full/ --dictionary currentDict.pkl --cuda --checkpoint voc62k_batch20_em1000_lay2_hid256_ep6_bptt20.mod --temperature 0.75 --outf ./outs/voc62k_batch20_em1000_lay2_hid256_ep6_bptt20_gen075.txt

python generate.py --data ./data/shortjokes_full/ --dictionary currentDict.pkl --cuda --checkpoint voc62k_batch20_em300_lay2_hid256_ep6_bptt20.mod  --temperature 0.5 --outf ./outs/voc62k_batch20_em300_lay2_hid256_ep6_bptt20_gen05.txt
python generate.py --data ./data/shortjokes_full/ --dictionary currentDict.pkl --cuda --checkpoint voc62k_batch20_em300_lay2_hid256_ep6_bptt20.mod  --temperature 0.75 --outf ./outs/voc62k_batch20_em300_lay2_hid256_ep6_bptt20_gen075.txt

python generate.py --data ./data/shortjokes_full/ --dictionary currentDict.pkl --cuda --checkpoint voc62k_batch20_em500_lay2_hid256_ep6_bptt20.mod  --temperature 0.5 --outf ./outs/voc62k_batch20_em500_lay2_hid256_ep6_bptt20_gen05.txt
python generate.py --data ./data/shortjokes_full/ --dictionary currentDict.pkl --cuda --checkpoint voc62k_batch20_em500_lay2_hid256_ep6_bptt20.mod  --temperature 0.75 --outf ./outs/voc62k_batch20_em500_lay2_hid256_ep6_bptt20_gen075.txt

python generate.py --data ./data/shortjokes_full/ --dictionary currentDict.pkl --cuda --checkpoint voc62k_batch20_em700_lay2_hid256_ep6_bptt20.mod  --temperature 0.5  --outf ./outs/voc62k_batch20_em700_lay2_hid256_ep6_bptt20_gen05.txt
python generate.py --data ./data/shortjokes_full/ --dictionary currentDict.pkl --cuda --checkpoint voc62k_batch20_em700_lay2_hid256_ep6_bptt20.mod  --temperature 0.75  --outf ./outs/voc62k_batch20_em700_lay2_hid256_ep6_bptt20_gen075.txt

python generate.py --data ./data/shortjokes_full/ --dictionary currentDict.pkl --cuda --checkpoint voc62k_batch15_em700_lay2_hid256_ep6_bptt35.mod --temperature 0.5  --outf ./outs/voc62k_batch15_em700_lay2_hid256_ep6_bptt35_gen05.txt
python generate.py --data ./data/shortjokes_full/ --dictionary currentDict.pkl --cuda --checkpoint voc62k_batch15_em700_lay2_hid256_ep6_bptt35.mod --temperature 0.75 --outf ./outs/voc62k_batch15_em700_lay2_hid256_ep6_bptt35_gen075.txt

python generate.py --data ./data/shortjokes_full/ --dictionary currentDict.pkl --cuda --checkpoint voc62k_batch10_em500_lay2_hid500_ep6_bptt20.mod --temperature 0.5  --outf  ./outs/voc62k_batch10_em500_lay2_hid500_ep6_bptt20_gen05.txt
python generate.py --data ./data/shortjokes_full/ --dictionary currentDict.pkl --cuda --checkpoint voc62k_batch10_em500_lay2_hid500_ep6_bptt20.mod --temperature 0.75  --outf  ./outs/voc62k_batch10_em500_lay2_hid500_ep6_bptt20_gen075.txt

python generate.py --data ./data/shortjokes_full/ --dictionary currentDict.pml --cuda --checkpoint voc62k_batch20_em500_lay2_hid500_ep6_bptt35.mod --temperature 0.5 --outf ./outs/voc62k_batch20_em500_lay2_hid500_ep6_bptt35_gen05.txt
python generate.py --data ./data/shortjokes_full/ --dictionary currentDict.pml --cuda --checkpoint voc62k_batch20_em500_lay2_hid500_ep6_bptt35.mod --temperature 0.75 --outf ./outs/voc62k_batch20_em500_lay2_hid500_ep6_bptt35_gen075.txt
