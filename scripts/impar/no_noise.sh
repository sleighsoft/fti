python impar.py -train ../raw_data/vgg-16/mnist_train_to_vgg-16_N[-1].npy -test ../raw_data/vgg-16/mnist_test_to_vgg-16_N[-1].npy -savefile ../impar/vgg-16/mnist/mnist -no_timestamp;
python impar.py -train ../raw_data/vgg-16/fashion_mnist_train_to_vgg-16_N[-1].npy -test ../raw_data/vgg-16/fashion_mnist_test_to_vgg-16_N[-1].npy -savefile ../impar/vgg-16/fashion_mnist/fashion_mnist -no_timestamp;
python impar.py -train ../raw_data/vgg-16/cifar10_train_to_vgg-16_N[-1].npy -test ../raw_data/vgg-16/cifar10_test_to_vgg-16_N[-1].npy -savefile ../impar/vgg-16/cifar10/cifar10 -no_timestamp;
python impar.py -train ../raw_data/vgg-16/cifar100_train_to_vgg-16_N[-1].npy -test ../raw_data/vgg-16/cifar100_test_to_vgg-16_N[-1].npy -savefile ../impar/vgg-16/cifar100/cifar100 -no_timestamp;
