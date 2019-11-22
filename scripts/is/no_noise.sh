# inception-v3
python is.py -data ../raw_data/inception-v3-final/mnist_train_to_inception-v3-final_N[-1].npy -savefile ../inception_score/inception-v3-final/mnist/mnist[train] -no_timestamp;
python is.py -data ../raw_data/inception-v3-final/mnist_test_to_inception-v3-final_N[-1].npy -savefile ../inception_score/inception-v3-final/mnist/mnist[test] -no_timestamp;
python is.py -data ../raw_data/inception-v3-final/fashion_mnist_train_to_inception-v3-final_N[-1].npy -savefile ../inception_score/inception-v3-final/fashion_mnist/fashion_mnist[train] -no_timestamp;
python is.py -data ../raw_data/inception-v3-final/fashion_mnist_test_to_inception-v3-final_N[-1].npy -savefile ../inception_score/inception-v3-final/fashion_mnist/fashion_mnist[test] -no_timestamp;
python is.py -data ../raw_data/inception-v3-final/cifar10_train_to_inception-v3-final_N[-1].npy -savefile ../inception_score/inception-v3-final/cifar10/cifar10[train] -no_timestamp;
python is.py -data ../raw_data/inception-v3-final/cifar10_test_to_inception-v3-final_N[-1].npy -savefile ../inception_score/inception-v3-final/cifar10/cifar10[test] -no_timestamp;
python is.py -data ../raw_data/inception-v3-final/cifar100_train_to_inception-v3-final_N[-1].npy -savefile ../inception_score/inception-v3-final/cifar100/cifar100[train] -no_timestamp;
python is.py -data ../raw_data/inception-v3-final/cifar100_test_to_inception-v3-final_N[-1].npy -savefile ../inception_score/inception-v3-final/cifar100/cifar100[test] -no_timestamp;