# MNIST
python -m data.tfds_to_activation.py \
../raw_data/ \
mnist \
-splits train test \
-frozen_network=vgg \
-frozen_network_version=16

# CIFAR10
python -m data.tfds_to_activation.py \
../raw_data/ \
cifar10 \
-splits train test \
-frozen_network=vgg \
-frozen_network_version=16

# FASHION-MNIST
python -m data.tfds_to_activation.py \
../raw_data/ \
fashion_mnist \
-splits train test \
-frozen_network=vgg \
-frozen_network_version=16

# CIFAR100
python -m data.tfds_to_activation.py \
../raw_data/ \
cifar100 \
-splits train test \
-frozen_network=vgg \
-frozen_network_version=16