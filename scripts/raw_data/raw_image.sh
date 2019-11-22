# MNIST
python tfds_to_numpy.py \
../raw_data/ \
mnist \
-splits train test

# CIFAR10
python tfds_to_numpy.py \
../raw_data/ \
cifar10 \
-splits train test

# CIFAR100
python tfds_to_numpy.py \
../raw_data/ \
cifar100 \
-splits train test

# FASHION-MNIST
python tfds_to_numpy.py \
../raw_data/ \
fashion_mnist \
-splits train test