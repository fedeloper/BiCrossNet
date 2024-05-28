name="bicrossnet_university"
data_dir="../CrossViewBNN/crossview/data/university1652/train"
test_dir="../CrossViewBNN/crossview/data/university1652/test"
gpu_ids="0"
lr=0.006
batchsize=8
triplet_loss=0.3
num_epochs=300
views=2
teacher="./university_tiny.pth"
optimizer='AdamLion' #['Adam', 'SGD', 'Lion','AdamLion'],
n1=20
n2=20
dataset="university"
python3 train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --views $views --lr $lr \
 --batchsize $batchsize --triplet_loss $triplet_loss --epochs $num_epochs --optimizer $optimizer --bigradual_n1 $n1 --bigradual_n2 $n2 --dataset $dataset --teacher_path=$teacher  \

for ((j = 1; j < 3; j++));
    do
      python3 test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --mode $j --model $name
    done
