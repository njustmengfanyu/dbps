# 生成干净数据集，只取前两个分类，所以干净预算400
python create_clean_set.py -dataset=cifar10 -clean_budget 600

# Create a poisoned training set.
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0 -cover_rate=0

python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0 -cover_rate=0

python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0 -cover_rate=0

python visualize.py -method=tsne -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0 -cover_rate=0

python ct_cleanser.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0 -cover_rate=0 -devices=0,1,2,3,4,5,6,7 -debug_info

