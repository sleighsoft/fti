# mnist (Blur)
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-0.25/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 0.25 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-0.50/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 0.50 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-0.75/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 0.75 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-1.00/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 1.00 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-1.50/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 1.50 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-2.00/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 2.00 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-3.00/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 3.00 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-4.00/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 4.00 -no_labels;

# mnist (Gaussian)
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0001/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0001 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0003/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0003 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0005/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0005 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0010/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0010 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0020/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0020 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0030/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0030 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0040/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0040 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0050/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0050 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0100/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0100 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0200/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0200 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0300/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0300 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0400/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0400 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0500/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0500 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.1000/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.1000 -no_labels;

# mnist (Salt & Pepper)
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.01/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.01 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.02/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.02 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.03/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.03 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.05/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.05 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.10/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.10 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.20/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.20 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.30/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.30 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.40/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.40 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.50/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.50 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-1.00/ mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 1.00 -no_labels;


# fashion_mnist (Blur)
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-0.25/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 0.25 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-0.50/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 0.50 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-0.75/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 0.75 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-1.00/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 1.00 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-1.50/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 1.50 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-2.00/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 2.00 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-3.00/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 3.00 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-4.00/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 4.00 -no_labels;

# fashion_mnist (Gaussian)
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0001/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0001 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0003/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0003 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0005/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0005 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0010/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0010 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0020/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0020 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0030/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0030 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0040/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0040 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0050/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0050 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0100/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0100 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0200/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0200 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0300/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0300 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0400/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0400 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0500/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0500 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.1000/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.1000 -no_labels;

# fashion_mnist (Salt & Pepper)
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.01/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.01 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.02/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.02 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.03/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.03 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.05/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.05 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.10/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.10 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.20/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.20 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.30/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.30 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.40/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.40 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.50/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.50 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-1.00/ fashion_mnist -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 1.00 -no_labels;



# CIFAR10 (Blur)
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-0.25/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 0.25 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-0.50/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 0.50 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-0.75/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 0.75 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-1.00/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 1.00 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-1.50/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 1.50 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-2.00/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 2.00 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-3.00/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 3.00 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-4.00/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 4.00 -no_labels;

# CIFAR10 (Gaussian)
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0001/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0001 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0003/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0003 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0005/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0005 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0010/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0010 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0020/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0020 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0030/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0030 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0040/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0040 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0050/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0050 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0100/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0100 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0200/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0200 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0300/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0300 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0400/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0400 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0500/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0500 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.1000/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.1000 -no_labels;

# CIFAR10 (Salt & Pepper)
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.01/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.01 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.02/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.02 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.03/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.03 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.05/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.05 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.10/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.10 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.20/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.20 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.30/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.30 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.40/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.40 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.50/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.50 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-1.00/ cifar10 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 1.00 -no_labels;


# CIFAR100 (Blur)
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-0.25/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 0.25 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-0.50/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 0.50 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-0.75/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 0.75 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-1.00/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 1.00 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-1.50/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 1.50 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-2.00/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 2.00 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-3.00/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 3.00 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/blur/blur-4.00/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "blur" -noise_amount 4.00 -no_labels;

# CIFAR100 (Gaussian)
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0001/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0001 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0003/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0003 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0005/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0005 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0010/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0010 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0020/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0020 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0030/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0030 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0040/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0040 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0050/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0050 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0100/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0100 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0200/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0200 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0300/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0300 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0400/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0400 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.0500/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.0500 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/gaussian/gaussian-0.1000/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "gaussian" -noise_amount 0.1000 -no_labels;

# CIFAR100 (Salt & Pepper)
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.01/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.01 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.02/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.02 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.03/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.03 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.05/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.05 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.10/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.10 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.20/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.20 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.30/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.30 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.40/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.40 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-0.50/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 0.50 -no_labels;
python -m data.tfds_to_activation /mnt/ssd/julian/raw_data/sap/sap-1.00/ cifar100 -splits test -frozen_network=inception -frozen_network_version=v3 -noise "sap" -noise_amount 1.00 -no_labels;
