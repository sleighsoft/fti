# Copyright 2019 Julian Niedermeier & Goncalo Mordido
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

import numba
import numpy as np
from sklearn.metrics import pairwise_distances


SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf


@numba.njit(parallel=True, cache=True)
def parallel_adaptive_knn_indices(X, n_neighbors):
    """This handles the case of multiple farthest neighbors that all have an
    equal distance to the center.
    
    Parameters
    ----------
    X: array of shape (n_samples, n_samples) or tuple with shape.
        A pairwise distance matrix.

    n_neighbors: int
        The number of neighbors to compute distances for.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.

    Returns
    -------
    knn_indices: array of shape (X[0], n_neighbors + e) where `e` is either 0 or
                 the number of identical farthest neighbors - 1.
        The indices of the k-nearest neighbors.
    knn_mask: array of shape (X[0], n_neighbors + e) where `e` is either 0 or
          the number of identical farthest neighbors - 1.
        A boolean mask that masks indices in knn_indices for points that do not
        have multiple farthest neighbors (for which `e=0`).
    """
    # We do not know `e` yet, so we have to assume the full shape
    knn_indices = np.empty(X.shape, dtype=np.int64)
    knn_mask = np.zeros_like(X, dtype=np.bool_)
    # Per point farthest neighbor index
    max_knn_indices = np.empty(X.shape[0], dtype=np.int64)
    for i in numba.prange(knn_indices.shape[0]):  # O(n_samples)
        row = X[i]
        # TODO Do not fully sort the array.
        #      Problem: no support for np.argpartition in numba
        #      https://github.com/numba/numba/issues/2445
        argsort = row.argsort(kind="quicksort")  # O(n_samples*log(n_samples))
        # Could be O(n_samples * n_neighbors * log(n_neighbors)) with argpartition + sort of partition
        farthest_knn_index = n_neighbors
        knn = row[argsort[n_neighbors - 1]]
        for j in range(
            farthest_knn_index, knn_indices.shape[1]
        ):  # O(n_samples) probably constant in reality
            if row[argsort[j]] == knn:
                farthest_knn_index += 1
            else:
                break
        knn_indices[i] = argsort
        knn_mask[i, :farthest_knn_index] = True
        max_knn_indices[i] = farthest_knn_index
    # max_knn_index = max_knn_indices.max()
    # return knn_indices[:, :max_knn_index], knn_mask[:, :max_knn_index]
    max_knn_index = max_knn_indices.max()
    knn_mask = knn_mask[:, :max_knn_index]
    knn_indices = knn_indices[:, :max_knn_index]
    return knn_indices, knn_mask


@numba.njit(parallel=True, cache=True)
def parallel_searchsorted(a, v):
    """Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted array a such that, if the corresponding
    elements in v were inserted before the indices, the order of a would be
    preserved.

    Assuming that a is sorted:

    ============================
    returned index `i` satisfies
    ============================
    ``a[i-1] < v <= a[i]``

    Parameters
    ----------
    a : 1-D array_like
        Input array. If `sorter` is None, then it must be sorted in
        ascending order, otherwise `sorter` must be an array of indices
        that sort it.
    v : array_like
        Values to insert into `a`.

    Returns
    -------
    indices : array of ints of shape (a[0].shape < v.shape)
        Array of insertion points with the same shape as `v`.
    """
    indices = np.empty(v.shape, dtype=np.int64)
    for i in numba.prange(v.shape[0]):
        indices[i] = np.searchsorted(a[i], v[i])
    return indices


@numba.njit(parallel=True, cache=True)
def parallel_sort_by_argsort(X, argsort_indices):
    """Sorts an array given a list of indices obtained from calling argsort."""
    sorted_array = np.empty(argsort_indices.shape, dtype=X.dtype)
    for i in numba.prange(sorted_array.shape[0]):
        sorted_array[i] = X[i][argsort_indices[i]]
    return sorted_array


@numba.njit(parallel=True, cache=True)
def parallel_take_along_axis(X, indices):
    """Creates a new array of indices.shape (must be 2-D) containing the values of X at indices."""
    taken = np.empty(indices.shape, dtype=X.dtype)
    for i in numba.prange(taken.shape[0]):
        row_indices = indices[i]
        row_X = X[i]
        for j in range(taken.shape[1]):
            taken[i][j] = row_X[row_indices[j]]
    return taken


@numba.njit(parallel=True, cache=True)
def parallel_put_by_advanced_index(X, advanced_indices, values):
    """Takes a 2-tuple of 1-D indices and puts a value at said indices."""
    x_indices, y_indices = advanced_indices
    for i in numba.prange(x_indices.shape[0]):
        x = x_indices[i]
        y = y_indices[i]
        X[x, y] = values[i]
    return X


@numba.njit(parallel=True, cache=True)
def parallel_put_by_advanced_index_scalar(X, advanced_indices, value):
    """Takes a 2-tuple of 1-D indices and puts the same value at said indices."""
    x_indices, y_indices = advanced_indices
    for i in numba.prange(x_indices.shape[0]):
        x = x_indices[i]
        y = y_indices[i]
        X[x, y] = value
    return X


@numba.njit(parallel=True, cache=True)
def parallel_take_by_advanced_index(X, advanced_indices):
    """Takes a 2-tuple of 1-D indices and returns the values at said indices."""
    x_indices, y_indices = advanced_indices
    new_array = np.empty(x_indices.shape[0], dtype=X.dtype)
    for i in numba.prange(new_array.shape[0]):
        x = x_indices[i]
        y = y_indices[i]
        new_array[i] = X[x, y]
    return new_array


@numba.njit(parallel=False, cache=True)
def insert_and_update_mask(
    knn_indices,
    knn_mask,
    knn_dists,
    index,
    insertion_indices,
    insertion_knn_dists,
    k,
    indices_shifted_by=1,
):
    """Allows updating an `parallel_adaptive_knn` result and its knn_mask. Inserts
    index at insertion_indices into knn_indices by shifting original values to
    the right. This method insures that the constraint of maintaining equally
    farthest neighbors holds and updates knn_mask accordingly.

    It does this by looking at new_knn_dists and old_knn_dists.

    Parameters
    ----------
    knn_indices : 2-D array of shape [N, M]
        k-NN indices.

    knn_mask : 2-D boolean array of shape [N, M]
        Mask corresponding to knn_indices.

    knn_dists : 2-D array of shape [N, M]
        Distances corresponding to knn_indices.

    index : scalar
        An index to insert into knn_indices.

    insertion_indices : 1-D array of shape [N]
        Precomputed positions where index will be inserted. Should always be
        left of index with identical distance.

    insertion_knn_dists : 1-D array of shape [N]
        Distances of index to row (point).

    k : int, whith k <= M
        Number of nearest neighbors used to compute knn_indices. Note, that this
        definition of k includes the center point itself.
        
        A -- B -- C -- D

        3-NN of A is C
        4-NN of A is D

    indices_shifted_by : int
        The amount of places knn_indices has been shifted. A positive value
        indicates a right shift.

    Returns
    -------
    new_knn_indices: array of shape (N, M+1)
        Updated knn_indices.

    new_knn_mask: array of shape (N, M+1)
        Updated knn_mask.
    """
    N = knn_indices.shape[0]
    M = knn_indices.shape[1]
    # Internally, we use the definition of k that does not inlcude
    k = k - 1
    # Both knn_indices and knn_mask grow by at most 1 column when index is
    # inserted.
    # shape = [N, M + 1]
    new_knn_indices = np.full((N, M + 1), -1, dtype=knn_indices.dtype)
    # shape = [N, M + 1]
    new_knn_mask = np.zeros((N, M + 1), dtype=knn_mask.dtype)
    # Copy knn_mask into new_knn_mask
    new_knn_mask[:, :-1] = knn_mask
    for i in numba.prange(N):
        # Where to put index
        pos = insertion_indices[i]
        # Row to be updated
        new_knn_indices_row = new_knn_indices[i]
        # Mask to be updated
        new_knn_mask_row = new_knn_mask[i]
        # Original knn indices
        knn_indices_row = knn_indices[i]
        # Distances matching original knn indices
        knn_dists_row = knn_dists[i]
        # Distance of index to center
        index_distance = insertion_knn_dists[i]
        if pos <= k:
            # Update indices
            new_knn_indices_row[pos] = index
            if pos != 0:
                # Part before pos
                new_knn_indices_row[:pos] = knn_indices_row[:pos]
            # Part after pos
            new_knn_indices_row[pos + 1 :] = knn_indices_row[pos:]

            # Update knn_mask
            old_farthest_neighbor_distance = knn_dists_row[k]
            if pos == k:
                # Index is the new farthest neighbor
                new_farthest_neighbor_distance = index_distance
            else:
                # Check what the new farthest neighbor distance is
                new_farthest_neighbor_distance = knn_dists_row[
                    new_knn_indices_row[k] - indices_shifted_by
                ]

            if old_farthest_neighbor_distance == new_farthest_neighbor_distance:
                # Farthest neighbor did not change. Mask grows by 1.
                # TODO argmin on np.bool currently does not support parallel=True
                #      Tracked in: https://github.com/numba/numba/issues/5263
                new_knn_mask_row[new_knn_mask_row.argmin()] = True
            else:
                # Farthest neighbor did change, shrink mask to k.
                new_knn_mask_row[k + 1 :] = False
        else:
            new_knn_indices_row[:-1] = knn_indices_row

    # TODO maybe trim array based on mask. check if performance increase.
    return new_knn_indices, new_knn_mask


# Original Author: Leland McInnes <leland.mcinnes@gmail.com>
# Published in: UMAP: Uniform Manifold Approximation and Projection
# License: BSD 3 clause
@numba.njit(fastmath=True, cache=True)
def smooth_knn_dist(
    distances, knn_mask, n_iter=64, local_connectivity=0.0, bandwidth=1.0
):
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In esscence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.

    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each samples. Each row should be a
        sorted list of distances to a given samples nearest neighbors.

    knn_mask: array of shape (n_samples, n_neighbors)
        A mask for knn_indices & knn_dists where False indicates not being
        a nearest neighbor.

    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.

    local_connectivity: int (optional, default 0)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.

    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated (sigma).

    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point (rho).
    """
    rho = np.zeros(distances.shape[0])
    sigma = np.zeros(distances.shape[0])

    mean_distances = np.mean(
        parallel_take_by_advanced_index(distances, np.where(knn_mask))
    )

    for i in range(distances.shape[0]):
        row_mask = knn_mask[i]
        row_k = np.maximum(
            (row_mask.argmin() == 0) * knn_mask.shape[1], row_mask.argmin()
        )
        target = np.log2(row_k) * bandwidth
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if (
            non_zero_dists.shape[0] > 0
            and non_zero_dists.shape[0] >= local_connectivity
        ):
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for _ in range(n_iter):

            psum = 0.0
            for j in range(1, row_k):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        sigma[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if sigma[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                sigma[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if sigma[i] < MIN_K_DIST_SCALE * mean_distances:
                sigma[i] = MIN_K_DIST_SCALE * mean_distances

    return sigma, rho


# Original Author: Leland McInnes <leland.mcinnes@gmail.com>
# Published in: UMAP: Uniform Manifold Approximation and Projection
# License: BSD 3 clause
@numba.njit(fastmath=True, cache=True)
def compute_membership_strengths(knn_indices, knn_dists, knn_mask, sigmas, rhos):
    """Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.

    Parameters
    ----------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.

    knn_mask: array of shape (n_samples, n_neighbors)
        A mask for knn_indices & knn_dists where False indicates not being
        a nearest neighbor.

    sigmas: array of shape(n_samples)
        The normalization factor derived from the metric tensor approximation.

    rhos: array of shape(n_samples)
        The local connectivity adjustment.

    Returns
    -------
    rows: array of shape (n_samples * n_neighbors)
        Row data for the resulting sparse matrix (coo format)

    cols: array of shape (n_samples * n_neighbors)
        Column data for the resulting sparse matrix (coo format)

    vals: array of shape (n_samples * n_neighbors)
        Entries for the resulting sparse matrix (coo format)
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    total_elements = np.sum(knn_mask)

    rows = np.zeros(total_elements, dtype=np.int64)
    cols = np.zeros(total_elements, dtype=np.int64)
    vals = np.zeros(total_elements, dtype=np.float64)

    offset = 0
    for i in range(n_samples):
        ith_mask = knn_mask[i]
        for j in range(n_neighbors):
            if ith_mask[j] == False:
                continue
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[offset + j] = i
            cols[offset + j] = knn_indices[i, j]
            vals[offset + j] = val
        offset += np.sum(ith_mask)

    return rows, cols, vals


# Original Author: Leland McInnes <leland.mcinnes@gmail.com>
# Published in: UMAP: Uniform Manifold Approximation and Projection
# License: BSD 3 clause
@numba.njit(fastmath=True, cache=True)
def fuzzy_simplicial_set(
    n_neighbors, knn_indices, knn_dists, knn_mask, local_connectivity=0.0
):
    """Given a neighborhood size, and a measure of distance compute the fuzzy
    simplicial set (here represented as a fuzzy graph in the form of a sparse
    matrix of rows, columns and values) associated to the data. This is done
    by locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point.

    Parameters
    ----------
    n_neighbors: int
        The number of neighbors to use to approximate geodesic distance.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.

    knn_indices: array of shape (n_samples, n_neighbors)
        The k-nearest neighbors of each point. This should be
        an array with the indices of the k-nearest neighbors as a row for
        each data point.

    knn_dists: array of shape (n_samples, n_neighbors)
        The k-nearest neighbors of each point. This should be
        an array with the distances of the k-nearest neighbors as a row for
        each data point.

    knn_mask: array of shape (n_samples, n_neighbors)
        A mask for knn_indices & knn_dists where False indicates not being
        a nearest neighbor.

    local_connectivity: int (optional, default 0)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    Returns
    -------
    fuzzy_simplicial_set: A 3-tuple rows, cols, vals
        A fuzzy simplicial set which can represented as a sparse matrix using
        rows, cols vals. The (i,j) entry of the matrix represents the directed
        membership strength of the 1-simplex between the ith and jth sample points.
    """
    sigmas, rhos = smooth_knn_dist(
        knn_dists, knn_mask, local_connectivity=local_connectivity
    )

    rows, cols, vals = compute_membership_strengths(
        knn_indices, knn_dists, knn_mask, sigmas, rhos
    )

    return rows, cols, vals


@numba.njit(parallel=True, fastmath=True, cache=True)
def fti(
    X_dmat,
    X_knn_mask,
    X_knn_dists,
    X_knn_indices_rshifted,
    X_knn_dists_xprime_insertion_indices,
    Xprime_to_X_dmat,  # TODO consider moving the concatenation of 0-column in here
    Xprime_to_X_knn_indices_rshifted,
    Xprime_to_X_knn_mask,
    k,
    local_connectivity,
):
    """Computes the fuzzy topology impact given precomputed values for a 
    reference dataset X and another dataset Xprime.

    Note k might be greater for some points than the initial k due to
    `parallel_adaptive_knn`. This is denoted by m >= k

    Parameters
    ----------
    X_dmat: float array of shape (n_samples, n_samples)
        Pairwise distance matrix of points in X.

    X_knn_mask: bool array of shape (n_samples, m)
        Masks out indices and distances from X_knn that are not valid. See note
        for `parallel_adaptive_knn`.

    X_knn_dists: float array of shape (n_samples, m)
        Distances of the k-nearest neighbors in X.

    X_knn_indices_rshifted: int array of shape (n_samples, m)
        Indices into the pairwise distance matrix of k-nearest neighbors in X.
        Shifted by 1 to the right to speed up computation later.

    X_knn_dists_xprime_insertion_indices: int array of shape (n_samples_2, n_samples)
        Insertion indices into X_knn_dists for points in Xprime.

    Xprime_to_X_dmat: float array of shape (n_samples_2, n_samples+1)
        Distances of points in Xprime to points in X. Last column has to be
        all 0.

    Xprime_to_X_knn_indices_rshifted: int array of shape (n_samples_2, m)
        Indices into the the pairwise distance matrix Xprime_to_X_dmat.
        Shifted by 1 to the right to speed up computation later.

    Xprime_to_X_knn_mask: bool array of shape (n_samples_2, m)
        Masks out indices and distances from Xprime_to_X_knn_indices_rshifted
        that are not valid. See note for `parallel_adaptive_knn`.

    k: int
        The k for k-nearest neighbor search. Note, that this definition of k
        will include the center point itself.

        A -- B -- C -- D

        3-NN of A is C
        4-NN of A is D

    local_connectivity: int (optional, default 0)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be set to 0.

    Returns
    -------
    P_Xprime: array of shape (n_samples_2)
        List of sum of edge probabilities of points in Xprime when added to X.
    """
    n_samples = X_dmat.shape[0]
    n_samples_2 = X_knn_dists_xprime_insertion_indices.shape[0]
    m = X_knn_mask.shape[1]

    assert k <= m

    assert X_knn_mask.shape == X_knn_dists.shape
    assert X_knn_mask.shape == X_knn_indices_rshifted.shape

    assert X_knn_dists_xprime_insertion_indices.shape[0] == n_samples_2
    assert X_knn_dists_xprime_insertion_indices.shape[1] == n_samples

    assert Xprime_to_X_dmat.shape[0] == n_samples_2
    assert Xprime_to_X_dmat.shape[1] == n_samples

    assert Xprime_to_X_knn_indices_rshifted.shape[0] == n_samples_2
    assert Xprime_to_X_knn_indices_rshifted.shape[1] == m

    assert Xprime_to_X_knn_indices_rshifted.shape == Xprime_to_X_knn_mask.shape

    # We append 0 distances, because we later left shift the indices after adding
    # the 0 index. This way we can map the knn_index of -1 to a distance of 0 in
    # a single pass.
    #
    # shape = [n_samples_2, n_samples+1]
    Xprime_to_X_dmat = np.concatenate(
        (
            Xprime_to_X_dmat,
            np.zeros((Xprime_to_X_dmat.shape[0], 1), dtype=Xprime_to_X_dmat.dtype),
        ),
        axis=1,
    )

    P_Xprime = np.empty(Xprime_to_X_dmat.shape[0], dtype=np.float64)
    for i in numba.prange(Xprime_to_X_dmat.shape[0]):
        # Indices of Xprime to points in **X**
        # Solution for Xprime: This is a list of indices to the k-nearest neighbors of Xprime
        # shape = [k] # TODO currently +1
        xprime_knn_indices = Xprime_to_X_knn_indices_rshifted[i]

        # shape = [k] # TODO currently +1
        xprime_knn_mask = Xprime_to_X_knn_mask[i]

        # shape = [n_samples+1]
        xprime_dists = Xprime_to_X_dmat[i]

        # Indices of where Xprime is located as a nearest neighbor of each Point in X
        # shape = [n_samples]
        xprime_insertion_indices = X_knn_dists_xprime_insertion_indices[i]

        # Insert the Xprime (index 0) at the correct position into our sort indices
        # Solution for X: This is a 2-D array of indices to the k-nearest neighbors
        #                 for each point in X.
        # shape = [n_samples, m+1]
        X_knn_indices_with_xprime, X_knn_mask_with_xprime = insert_and_update_mask(
            X_knn_indices_rshifted,
            X_knn_mask,
            X_knn_dists,
            0,
            xprime_insertion_indices,
            xprime_dists,
            k,
        )

        # Solution: knn_indices
        #
        # [xprime_knn_indices,
        #  X_knn_indices_with_xprime]
        #
        # shape = [n_samples+1, m+1]
        knn_indices = np.zeros(
            (n_samples + 1, X_knn_indices_with_xprime.shape[1]),
            dtype=X_knn_indices_with_xprime.dtype,
        )

        knn_indices[0, : xprime_knn_indices.shape[0]] = xprime_knn_indices
        knn_indices[1:, :] = X_knn_indices_with_xprime

        # Solution: knn_mask
        #
        # [xprime_knn_mask,
        #  X_knn_mask_with_xprime]
        #
        # shape = [n_samples+1, m+1]
        knn_mask = np.zeros(
            (n_samples + 1, X_knn_mask_with_xprime.shape[1]),
            dtype=X_knn_mask_with_xprime.dtype,
        )

        knn_mask[0, : xprime_knn_mask.shape[0]] = xprime_knn_mask
        knn_mask[1:, :] = X_knn_mask_with_xprime

        # We have to left shift the indices to use them to index into Xprime_to_X_dmat
        # This will turn the previously added 0 index into -1.
        # We can do this inplace because we will not touch that row again.
        np.subtract(xprime_knn_indices, 1, xprime_knn_indices)

        # Distance to itself is 0 due to prepend 0 to Xprime_to_X_dmat trick.
        # Solution Xprime: This is a list of distances to the k-nearest neighbors of Xprime.
        xprime_knn_dists = xprime_dists[xprime_knn_indices]

        # We have to left shift indices back to use them as indices into X_dmat.
        # We can do this inplace because we will not touch the indices again
        np.subtract(X_knn_indices_with_xprime, 1, X_knn_indices_with_xprime)

        # Get the distances from X_dmat
        X_knn_dists_with_xprime = parallel_take_along_axis(
            X_dmat, X_knn_indices_with_xprime
        )

        # Due to the left shift the 0 index of our Xprime is now -1. We therefore determine
        # all indices where it is -1 and use them to select the correct distances from
        # Xprime_to_X_dmat instead of X_dmat.
        new_point_indices = np.where(X_knn_indices_with_xprime == -1)

        # Get the distances from Xprime_to_X_dmat and override incorrect values in
        # X_knn_dists_with_xprime.
        # Solution X: This is a 2-D array of distances to the k-nearest neighbors for
        #             each point in X.
        parallel_put_by_advanced_index(
            X_knn_dists_with_xprime,
            new_point_indices,
            Xprime_to_X_dmat[i][new_point_indices[0]],
        )

        # Solution: knn_dists
        knn_dists = np.zeros(
            (n_samples + 1, X_knn_dists_with_xprime.shape[1]),
            dtype=X_knn_dists_with_xprime.dtype,
        )

        knn_dists[0, 0 : xprime_knn_dists.shape[0]] = xprime_knn_dists
        knn_dists[1:, :] = X_knn_dists_with_xprime

        # TODO It might be possible to omit this and do it in smooth_knn_dist instead
        # Set Mask == False to -1 for fuzzy_simplical_set
        masked_indices = np.where(knn_mask == np.array(False, dtype=np.bool_))
        parallel_put_by_advanced_index_scalar(knn_dists, masked_indices, -1)

        # This is basically fs-set_X_xprime
        rows, cols, vals = fuzzy_simplicial_set(
            n_neighbors=k,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
            knn_mask=knn_mask,
            local_connectivity=local_connectivity,
        )

        # rows, cols, vals describe a sparse matrix, we only want the dense values (vals == 0).
        # We also do not want Xprime's edges (rows == 0, cols == 0).
        view_indices = np.where(
            np.logical_and(np.logical_and(rows != 0, cols != 0), vals != 0)
        )[0]
        fs_set_X_Xprime_minus_xprime = vals[view_indices]
        P_X_Xprime_minus_xprime = np.sum(fs_set_X_Xprime_minus_xprime)
        P_Xprime[i] = P_X_Xprime_minus_xprime

    return P_Xprime


def fuzzy_topology_impact(
    X, Xprime, k=4, metric="euclidean", local_connectivity=0.0, X_dmat=None
):
    """Computes the fuzzy topology impact given a reference dataset X and another
    dataset Xprime.

    Precomputes several values to speed up computation of FTI.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        Reference data.

    Xprime: array of shape (n_samples_2, n_features)
        Other data.

    k: int
        The k for k-nearest neighbor search. Note, that this definition of k
        will include the center point itself.
        
        A -- B -- C -- D

        3-NN of A is C
        4-NN of A is D

    metric : string, or callable (optional, default euclidean)
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    local_connectivity: int (optional, default 0)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be set to 0.

    X_dmat: array of shape (n_samples, n_samples) (optional, default None)
        A precomputed pairwise distance matrix of X.

    Returns
    -------
    impact: float
        The fuzzy topology impact given X and Xprime.
    """
    X = X.astype(np.float64, copy=False)
    Xprime = Xprime.astype(np.float64, copy=False)

    # TODO Replace pairwise_distances with a faster numba version
    if X_dmat is None:
        # Compute pairwise distance matrix for REFERENCE
        #
        # shape = [n_samples, n_samples]
        X_dmat = pairwise_distances(X, metric=metric)

    # Compute k-NN indices for REFERENCE
    #
    # shape = [n_samples, k+e1], where e1 >= 0
    X_knn_indices, X_knn_mask = parallel_adaptive_knn(X_dmat, k)

    # Get k-NN distances for REFERENCE
    #
    # shape = [n_samples, k+e1], where e1 >= 0
    # TODO Copy might not be necessary
    X_knn_dists = X_dmat[np.arange(X_dmat.shape[0])[:, None], X_knn_indices].copy()
    # Set distances that are masked out to -1
    X_knn_dists[np.logical_not(X_knn_mask)] = -1

    _, _, fs_set_X = fuzzy_simplicial_set(
        k, X_knn_indices, X_knn_dists, X_knn_mask, local_connectivity
    )

    P_X = np.sum(fs_set_X)

    # As the center node itself is at index 0, but is not included in our computation,
    # we have to increment all indices by 1.
    # If train dmat = [5 x 5] it would be [6 x 6] with Xprime prepended
    # We therefore shift the [5 x 5] indices by 1 to the right
    #  __         __
    # |0 _        _|
    # | |1 2 3 4 5 |
    # | |2         |
    # | |3         |
    # | |4         |
    # |_|5        _|
    np.add(X_knn_indices, 1, out=X_knn_indices)
    # Renaming is just for clarity
    X_knn_indices_rshifted = X_knn_indices

    # Compute pairwise distance matrix between REFERENCE - OTHER
    # => For every point in Xprime, compute how far it is from every point in X
    # Row = Xprime, Col = Distance to Train Point
    #
    # shape = [n_samples_2, n_samples]
    # TODO Replace pairwise_distances with a faster numba version
    Xprime_to_X_dmat = pairwise_distances(Xprime, X, metric=metric)

    # Compute where OTHER points lie within the k-NN of REFERENCE
    # Row = Point, Column = Where to insert
    #
    # shape = [n_samples_2, n_samples]
    X_knn_dists_xprime_insertion_indices = parallel_searchsorted(
        X_knn_dists, Xprime_to_X_dmat.T
    ).T

    # Compute k-NN indices for OTHER to REFERENCE.
    #
    # We set k-1 as we later include the center point as the closest neighbor.
    #
    # Row = Xprime, Col = Index into dmat
    #
    # We now have:
    # [[1-NN, 2-NN, ..., (k-1+e2)-NN],
    #  [5, 3, ..., (k-1+e2)-NN],         // example
    #  [...],
    #  [1-NN, 2-NN, ..., (k-1+e2)-NN]]
    #
    # shape = [n_samples_2, k-1+e2], where e2 >= 0
    Xprime_to_X_knn_indices, Xprime_to_X_knn_mask = parallel_adaptive_knn(
        Xprime_to_X_dmat, k - 1
    )

    # Shift by 1 due to prepend [see X_knn_indices_rshifted]
    # We now have:
    # [[1-NN, 2-NN, ..., (k-1+e2)-NN],
    #  [6, 4, ..., (k-1+e)-NN],         // example
    #  [...],
    #  [1-NN, 2-NN, ..., (k-1+e2)-NN]]
    np.add(Xprime_to_X_knn_indices, 1, out=Xprime_to_X_knn_indices)
    # Renaming is just for clarity
    Xprime_to_X_knn_indices_rshifted = Xprime_to_X_knn_indices

    # Because our interpretation of k-NN includes the center itself we have to
    # prepend the center as the point with the closest distance (index 0)
    #
    # We now have:
    # [[0, 1-NN, 2-NN, ..., (k+e2)-NN],
    #  [0, 5, 3, ..., (k+e2)-NN],         // example
    #  [...],
    #  [0, 1-NN, 2-NN, ..., (k+e2)-NN]]
    #
    # shape = [n_samples_2, k+e2], where e2 >= 0
    Xprime_to_X_knn_indices_rshifted = np.concatenate(
        (
            np.zeros(
                (Xprime_to_X_knn_indices_rshifted.shape[0], 1),
                dtype=Xprime_to_X_knn_indices_rshifted.dtype,
            ),
            Xprime_to_X_knn_indices_rshifted,
        ),
        axis=-1,
    )

    # We have to prepend a True to our mask, because we prepended the center point
    Xprime_to_X_knn_mask = np.concatenate(
        (
            np.ones(
                (Xprime_to_X_knn_mask.shape[0], 1), dtype=Xprime_to_X_knn_mask.dtype
            ),
            Xprime_to_X_knn_mask,
        ),
        axis=1,
    )

    # Handle the case where k+e1 < k+e2.
    # This can happen when a point in Xprime has more than k+e1 neighbors in X.
    padding = (
        Xprime_to_X_knn_indices_rshifted.shape[1] - X_knn_indices_rshifted.shape[1]
    )
    if padding > 0:
        # Simply expand all arrays with zeros and the mask with False
        X_knn_dists = np.concatenate(
            (
                X_knn_dists,
                np.zeros((X_knn_dists.shape[0], padding), dtype=X_knn_dists.dtype),
            ),
            axis=1,
        )
        X_knn_mask = np.concatenate(
            (
                X_knn_mask,
                np.zeros((X_knn_mask.shape[0], padding), dtype=X_knn_mask.dtype),
            ),
            axis=1,
        )
        X_knn_indices_rshifted = np.concatenate(
            (
                X_knn_indices_rshifted,
                np.zeros(
                    (X_knn_indices_rshifted.shape[0], padding),
                    dtype=X_knn_indices_rshifted.dtype,
                ),
            ),
            axis=1,
        )
    elif padding < 0:
        padding = abs(padding)
        # Simply expand all arrays with zeros and the mask with False
        Xprime_to_X_knn_mask = np.concatenate(
            (
                Xprime_to_X_knn_mask,
                np.zeros(
                    (Xprime_to_X_knn_mask.shape[0], padding),
                    dtype=Xprime_to_X_knn_mask.dtype,
                ),
            ),
            axis=1,
        )
        Xprime_to_X_knn_indices_rshifted = np.concatenate(
            (
                Xprime_to_X_knn_indices_rshifted,
                np.zeros(
                    (Xprime_to_X_knn_indices_rshifted.shape[0], padding),
                    dtype=Xprime_to_X_knn_indices_rshifted.dtype,
                ),
            ),
            axis=1,
        )
    P_X_Xprime_minus_xprime = fti(
        X_dmat,
        X_knn_mask,
        X_knn_dists,
        X_knn_indices_rshifted,
        X_knn_dists_xprime_insertion_indices,
        Xprime_to_X_dmat,
        Xprime_to_X_knn_indices_rshifted,
        Xprime_to_X_knn_mask,
        k,
        local_connectivity,
    )
    impact = (P_X - np.mean(P_X_Xprime_minus_xprime)) / fs_set_X.size
    return impact, P_X, P_X_Xprime_minus_xprime, fs_set_X.size
