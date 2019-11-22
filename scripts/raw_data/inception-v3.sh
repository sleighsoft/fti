# MNIST
python -m data.tfds_to_activation.py \
../raw_data/ \
mnist \
-splits train test \
-frozen_network=inception \
-frozen_network_version=v3

# CIFAR10
python -m data.tfds_to_activation.py \
../raw_data/ \
cifar10 \
-splits train test \
-frozen_network=inception \
-frozen_network_version=v3

# FASHION-MNIST
python -m data.tfds_to_activation.py \
../raw_data/ \
fashion_mnist \
-splits train test \
-frozen_network=inception \
-frozen_network_version=v3

# CIFAR100
python -m data.tfds_to_activation.py \
../raw_data/ \
cifar100 \
-splits train test \
-frozen_network=inception \
-frozen_network_version=v3