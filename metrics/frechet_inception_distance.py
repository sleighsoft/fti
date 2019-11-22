import numpy as np
from scipy import linalg


# Copyright 2019 The TensorFlow GAN Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def frechet_inception_distance(real, fake):
    """Computes the Fréchet Inception Distance as reported by https://arxiv.org/pdf/1706.08500.pdf.

    Code at: https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/eval/classifier_metrics.py#L1024

    Can be used with any input.

    Note: This does not run an inception network on real & fake!
    
    Args:
        real: Real data of shape [Batch_Real, N].
        fake: Fake data of shape [Batch_Fake, N].
    
    Returns:
        A scalar, the Fréchet Inception Distance.
    """
    assert np.ndim(real) == 2, "Dimension of real must be 2-d"
    assert np.ndim(fake) == 2, "Dimension of fake must be 2-d"

    dtype = real.dtype

    real = real.astype(np.float64, copy=False)
    fake = fake.astype(np.float64, copy=False)

    m = np.mean(real, axis=0)
    m_w = np.mean(fake, axis=0)

    # Calculate the unbiased covariance matrix of real_activations.
    num_examples_real = float(real.shape[0])
    sigma = (
        num_examples_real / (num_examples_real - 1) * np.cov(real, rowvar=False, ddof=0)
    )
    # Calculate the unbiased covariance matrix of generated_activations.
    num_examples_fake = float(fake.shape[0])
    sigma_w = (
        num_examples_fake / (num_examples_fake - 1) * np.cov(fake, rowvar=False, ddof=0)
    )

    # Find the Tr(sqrt(sigma sigma_w)) component of FID
    covmean = linalg.sqrtm(sigma_w.dot(sigma), disp=False)[0]

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    sqrt_trace_component = np.trace(covmean)

    # Compute the two components of FID.

    # First the covariance component.
    # Here, note that trace(A + B) = trace(A) + trace(B)
    trace = np.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    # Next the distance between means.
    diff = m - m_w
    mean = diff.dot(diff)  # Equivalent to L2 but more stable.
    fid = trace + mean
    fid = fid.astype(dtype, copy=False)
    return fid
