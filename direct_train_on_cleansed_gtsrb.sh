nohup python -u train_on_cleansed_set.py -dataset gtsrb -poison_type badnet -poison_rate 0.01 -devices 8 -seed 2333 > logs/gtsrb/train_on_cleansed/2333/badnet.out 2>&1 &
nohup python -u train_on_cleansed_set.py -dataset gtsrb -poison_type blend -poison_rate 0.01 -devices 9 -seed 2333 > logs/gtsrb/train_on_cleansed/2333/blend.out 2>&1 &
nohup python -u train_on_cleansed_set.py -dataset gtsrb -poison_type dynamic -poison_rate 0.01 -devices 2 -seed 2333 > logs/gtsrb/train_on_cleansed/2333/dynamic.out 2>&1 &
nohup python -u train_on_cleansed_set.py -dataset gtsrb -poison_type SIG -poison_rate 0.02 -devices 3 -seed 2333 > logs/gtsrb/train_on_cleansed/2333/SIG.out 2>&1 &
nohup python -u train_on_cleansed_set.py -dataset gtsrb -poison_type TaCT -poison_rate 0.02 -cover_rate 0.01 -devices 4 -seed 2333 > logs/gtsrb/train_on_cleansed/2333/TaCT.out 2>&1 &
nohup python -u train_on_cleansed_set.py -dataset gtsrb -poison_type adaptive_blend -poison_rate 0.003 -cover_rate 0.003 -alpha 0.15 -test_alpha 0.2 -devices 5 -seed 2333 > logs/gtsrb/train_on_cleansed/2333/adaptive_blend.out 2>&1 &
nohup python -u train_on_cleansed_set.py -dataset gtsrb -poison_type adaptive_patch -poison_rate 0.005 -cover_rate 0.01 -devices 6 -seed 2333 > logs/gtsrb/train_on_cleansed/2333/adaptive_patch.out 2>&1 &
nohup python -u train_on_cleansed_set.py -dataset gtsrb -poison_type none -devices 7 -seed 2333 > logs/gtsrb/train_on_cleansed/2333/none.out 2>&1 &