# download tar files
wget http://maxwell.cs.umass.edu/mvcnn-data/modelnet40v1.tar

# extract tar files
tar xvf modelnet40v1.tar

mkdir ./modelnet40
mkdir ./modelnet40/train
mkdir ./modelnet40/test

for c in $(ls ./modelnet40v1/)
do
   mkdir ./modelnet40/train/$c
   mv ./modelnet40v1/$c/train/* ./modelnet40/train/$c/
   mkdir ./modelnet40/test/$c
   mv ./modelnet40v1/$c/test/* ./modelnet40/test/$c/
done

rm -rf ./modelnet40v1
rm -rf modelnet40v1.tar
