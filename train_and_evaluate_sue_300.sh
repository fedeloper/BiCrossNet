name="bicrossnet_sue_300"
data_dir="../CrossViewBNN/crossview/data/sue/Training/300"
test_dir="../CrossViewBNN/crossview/data/sue/Testing/300"
gpu_ids="0"
lr=0.006
batchsize=8
teacher="./convtri_sue200.pth"
triplet_loss=0.3
num_epochs=300
views=2
optimizer="Adam-Lion" #['Adam', 'SGD', 'Lion','AdamLion'],
n1=20
n2=20
python3 train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --views $views --lr $lr \
 --batchsize $batchsize --triplet_loss $triplet_loss --epochs $num_epochs --optimizer $optimizer --bigradual_n1 $n1 --bigradual_n2 $n2  --teacher_path=$teacher \

for ((j = 1; j < 3; j++));
    do
      python3 test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --mode $j --model $name
    done
