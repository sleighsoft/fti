# raw
python fti.py -train ../raw_data/mnist_train_N[-1].npy -test ../raw_data/mnist_test_N[-1].npy -savefile ../fti_k20/mnist/mnist -no_timestamp -k 20;
python fti.py -test ../raw_data/mnist_train_N[-1].npy -train ../raw_data/mnist_test_N[-1].npy -savefile ../fti_k20/mnist/mnist.swapped -no_timestamp -k 20;
python fti.py -train ../raw_data/fashion_mnist_train_N[-1].npy -test ../raw_data/fashion_mnist_test_N[-1].npy -savefile ../fti_k20/fashion_mnist/fashion_mnist -no_timestamp -k 20;
python fti.py -test ../raw_data/fashion_mnist_train_N[-1].npy -train ../raw_data/fashion_mnist_test_N[-1].npy -savefile ../fti_k20/fashion_mnist/fashion_mnist.swapped -no_timestamp -k 20;
python fti.py -train ../raw_data/cifar10_train_N[-1].npy -test ../raw_data/cifar10_test_N[-1].npy -savefile ../fti_k20/cifar10/cifar10 -no_timestamp -k 20;
python fti.py -test ../raw_data/cifar10_train_N[-1].npy -train ../raw_data/cifar10_test_N[-1].npy -savefile ../fti_k20/cifar10/cifar10.swapped -no_timestamp -k 20;
python fti.py -train ../raw_data/cifar100_train_N[-1].npy -test ../raw_data/cifar100_test_N[-1].npy -savefile ../fti_k20/cifar100/cifar100 -no_timestamp -k 20;
python fti.py -test ../raw_data/cifar100_train_N[-1].npy -train ../raw_data/cifar100_test_N[-1].npy -savefile ../fti_k20/cifar100/cifar100.swapped -no_timestamp -k 20;
# vgg-16
python fti.py -train ../raw_data/vgg-16/mnist_train_to_vgg-16_N[-1].npy -test ../raw_data/vgg-16/mnist_test_to_vgg-16_N[-1].npy -savefile ../fti_k20/vgg-16/mnist/mnist -no_timestamp -k 20;
python fti.py -test ../raw_data/vgg-16/mnist_train_to_vgg-16_N[-1].npy -train ../raw_data/vgg-16/mnist_test_to_vgg-16_N[-1].npy -savefile ../fti_k20/vgg-16/mnist/mnist.swapped -no_timestamp -k 20;
python fti.py -train ../raw_data/vgg-16/fashion_mnist_train_to_vgg-16_N[-1].npy -test ../raw_data/vgg-16/fashion_mnist_test_to_vgg-16_N[-1].npy -savefile ../fti_k20/vgg-16/fashion_mnist/fashion_mnist -no_timestamp -k 20;
python fti.py -test ../raw_data/vgg-16/fashion_mnist_train_to_vgg-16_N[-1].npy -train ../raw_data/vgg-16/fashion_mnist_test_to_vgg-16_N[-1].npy -savefile ../fti_k20/vgg-16/fashion_mnist/fashion_mnist.swapped -no_timestamp -k 20;
python fti.py -train ../raw_data/vgg-16/cifar10_train_to_vgg-16_N[-1].npy -test ../raw_data/vgg-16/cifar10_test_to_vgg-16_N[-1].npy -savefile ../fti_k20/vgg-16/cifar10/cifar10 -no_timestamp -k 20;
python fti.py -test ../raw_data/vgg-16/cifar10_train_to_vgg-16_N[-1].npy -train ../raw_data/vgg-16/cifar10_test_to_vgg-16_N[-1].npy -savefile ../fti_k20/vgg-16/cifar10/cifar10.swapped -no_timestamp -k 20;
python fti.py -train ../raw_data/vgg-16/cifar100_train_to_vgg-16_N[-1].npy -test ../raw_data/vgg-16/cifar100_test_to_vgg-16_N[-1].npy -savefile ../fti_k20/vgg-16/cifar100/cifar100 -no_timestamp -k 20;
python fti.py -test ../raw_data/vgg-16/cifar100_train_to_vgg-16_N[-1].npy -train ../raw_data/vgg-16/cifar100_test_to_vgg-16_N[-1].npy -savefile ../fti_k20/vgg-16/cifar100/cifar100.swapped -no_timestamp -k 20;
# inception-v3
python fti.py -train ../raw_data/inception-v3/mnist_train_to_inception-v3_N[-1].npy -test ../raw_data/inception-v3/mnist_test_to_inception-v3_N[-1].npy -savefile ../fti_k20/inception-v3/mnist/mnist -no_timestamp -k 20;
python fti.py -test ../raw_data/inception-v3/mnist_train_to_inception-v3_N[-1].npy -train ../raw_data/inception-v3/mnist_test_to_inception-v3_N[-1].npy -savefile ../fti_k20/inception-v3/mnist/mnist.swapped -no_timestamp -k 20;
python fti.py -train ../raw_data/inception-v3/fashion_mnist_train_to_inception-v3_N[-1].npy -test ../raw_data/inception-v3/fashion_mnist_test_to_inception-v3_N[-1].npy -savefile ../fti_k20/inception-v3/fashion_mnist/fashion_mnist -no_timestamp -k 20;
python fti.py -test ../raw_data/inception-v3/fashion_mnist_train_to_inception-v3_N[-1].npy -train ../raw_data/inception-v3/fashion_mnist_test_to_inception-v3_N[-1].npy -savefile ../fti_k20/inception-v3/fashion_mnist/fashion_mnist.swapped -no_timestamp -k 20;
python fti.py -train ../raw_data/inception-v3/cifar10_train_to_inception-v3_N[-1].npy -test ../raw_data/inception-v3/cifar10_test_to_inception-v3_N[-1].npy -savefile ../fti_k20/inception-v3/cifar10/cifar10 -no_timestamp -k 20;
python fti.py -test ../raw_data/inception-v3/cifar10_train_to_inception-v3_N[-1].npy -train ../raw_data/inception-v3/cifar10_test_to_inception-v3_N[-1].npy -savefile ../fti_k20/inception-v3/cifar10/cifar10.swapped -no_timestamp -k 20;
python fti.py -train ../raw_data/inception-v3/cifar100_train_to_inception-v3_N[-1].npy -test ../raw_data/inception-v3/cifar100_test_to_inception-v3_N[-1].npy -savefile ../fti_k20/inception-v3/cifar100/cifar100 -no_timestamp -k 20;
python fti.py -test ../raw_data/inception-v3/cifar100_train_to_inception-v3_N[-1].npy -train ../raw_data/inception-v3/cifar100_test_to_inception-v3_N[-1].npy -savefile ../fti_k20/inception-v3/cifar100/cifar100.swapped -no_timestamp -k 20;