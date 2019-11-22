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


@numba.njit(cache=True)
def insert_and_update_mask(
    knn_indices,
    knn_mask,
    knn_dists,
    index,
    insertion_indices,
    insertion_knn_dists,
    n_neighbors,
):
    """Allows updating an adative knn_indices result and its knn_mask. Inserts index at insertion_indices
    into knn_indices by shifting original values to the right. This method insures that the
    constrained of maintaining equally farthest neighbors holds and updates knn_mask accordingly.

    It does this by looking at new_knn_dists and old_knn_dists.

    Parameters
    ----------
    knn_indices : 2-D array of shape [N, n_neighbors]
        k-NN indices.
    insertion_indices : 1-D array of shape [N]
        Indices where index will be inserted.
    index : scalar
        An index to insert into knn_indices.
    knn_mask : 2-D boolean array of same shape as knn_indices
        Mask corresponding to knn_indices.
    insertion_knn_dists : 1-D array of shape [N]
        Distances of index to row (point).
    knn_dists : 1-D array of same shape as knn_indices
        Distances corresponding to knn_indices.
    n_neighbors : scalar (int)
        Number of nearest neighbors.


    Returns
    -------
    new_knn_indices: array of shape (knn_indices[0], knn_indices[1]+1)
        Updated knn_indices.
    new_knn_mask: array of shape (knn_indices[0], knn_indices[1]+1)
        Updated knn_mask.
    """
    # Both the knn_indices and knn_mask grow by at most 1 column
    new_knn_indices = np.full(
        (knn_indices.shape[0], knn_indices.shape[1] + 1), -1, dtype=knn_indices.dtype
    )
    new_knn_mask = np.zeros(
        (knn_indices.shape[0], knn_indices.shape[1] + 1), dtype=knn_mask.dtype
    )
    new_knn_mask[:, :-1] = knn_mask
    # TODO Currently does not support parallel
    for i in numba.prange(new_knn_indices.shape[0]):
        pos = insertion_indices[i]
        new_knn_indices_row = new_knn_indices[i]
        new_knn_mask_row = new_knn_mask[i]
        knn_indices_row = knn_indices[i]
        knn_dists_row = knn_dists[i]
        if pos < knn_indices.shape[1]:
            new_knn_indices_row[pos] = index
            if pos != 0:
                # Part before pos
                new_knn_indices_row[:pos] = knn_indices_row[:pos]
            # Part after pos
            new_knn_indices_row[pos + 1 :] = knn_indices_row[pos:]
            # Update knn_mask (3 cases)
            # 1. Inserted point's distance is identical to farthest neighbor
            #    => True knn_mask grows by 1
            #    - Example: 5 points, k=4 (so 3 closest neighbors excluding itself)
            #      A|B    C D E
            #      C -> D = 1
            #      C -> E = 2
            #      C -> A&B = 3
            #
            #      KNN_INDICES[C] = C D E A B
            #      KNN_MASK[C]    = T T T T T
            #
            #   -> Now insert point F at distance = 3 to C
            #      KNN_INDICES[C] = C D E F A B
            #      KNN_MASK[C]    = T T T T T T
            #
            # 2. Inserted point's distance is not identical to farthest neighbor
            #    BUT ... multiple "identical farthest neighbors" exist
            #    2.1 new_knn_indices farthest neighbor
            #        => True knn_mask shrinks to k
            #        - Example: 5 points, k=4 (so 3 closest neighbors excluding itself)
            #          A|B    C D E
            #          C -> D = 1
            #          C -> E = 2
            #          C -> A&B = 3
            #
            #          KNN_INDICES[C] = C D E A B
            #          KNN_MASK[C]    = T T T T T
            #
            #       -> Now insert point F at a distance < 3 to C, e.g. 1.5
            #          => Now E!=A|B
            #          KNN_INDICES[C] = C D F E A B
            #          KNN_MASK[C]    = T T T T F F
            #    2.2 Farthest neighbor is still identical to previous farthest neighbor
            #        => True knn_mask grows by 1
            #        - Example: 5 points, k=5 (so 3 closest neighbors excluding itself)
            #          A|B    C D|E
            #          C -> D&E = 2
            #          C -> A&B = 3
            #
            #          KNN_INDICES[C] = C D E A B
            #          KNN_MASK[C]    = T T T T T
            #
            #       -> Now insert point F at a distance < 3 to C, e.g. 1.5
            #          KNN_INDICES[C] = C F D E A B
            #          KNN_MASK[C]    = T T T T T T
            # 3. Else
            #    => No change of True knn_mask necessary
            #    - Example: 4 points, k=4 (so 3 closest neighbors excluding itself)
            #      A    C D E
            #      C -> D = 1
            #      C -> E = 2
            #      C -> A = 3
            #
            #      KNN_INDICES[C] = C D E A
            #      KNN_MASK[C]    = T T T T
            #
            #   -> Now insert point F at a distance < 3 to C, e.g. 1.5
            #      KNN_INDICES[C] = C D F E
            #      KNN_MASK[C]    = T T T T
            farthest_neighbor_index = new_knn_mask_row.argmin()
            if knn_dists_row[farthest_neighbor_index - 1] == insertion_knn_dists[i]:
                # 1.
                new_knn_mask_row[farthest_neighbor_index] = True
            elif (
                knn_dists_row[farthest_neighbor_index - 1]
                == knn_dists_row[farthest_neighbor_index - 2]
            ):
                # 2.
                if (n_neighbors - 1 == insertion_indices[i]) or (
                    knn_dists_row[
                        n_neighbors - 1 - (insertion_indices[i] < n_neighbors - 1)
                    ]
                    != knn_dists_row[n_neighbors - 1]
                ):
                    # 2.1
                    new_knn_mask_row[n_neighbors:] = False
                else:
                    # 2.2
                    new_knn_mask_row[farthest_neighbor_index] = True
        else:
            new_knn_indices_row[:-1] = knn_indices_row
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
    X_knn_dists,
    X_knn_indices_rshifted,
    X_knn_dists_xprime_insertion_indices,
    X_knn_mask,
    Xprime_to_X_dmat,
    Xprime_to_X_knn_indices_rshifted,
    Xprime_to_X_knn_mask,
    k,
    local_connectivity,
):
    """Computes the fuzzy topology impact given precomputed values for a 
    reference dataset X and another dataset Xprime.

    Note k might be greater for some points than the initial k due to
    AdapativeKNearestNeighbors.

    Parameters
    ----------
    X_dmat: float array of shape (n_samples, n_samples)
        Pairwise distance matrix of points in X.

    X_knn_dists: float array of shape (n_samples, k)
        Distances of the k-nearest neighbors in X.

    X_knn_indices: int array of shape (n_samples, k)
        Indices into the pairwise distance matrix of k-nearest neighbors in X.
        Shifted by 1 to the right to speed up computation later.

    X_knn_dists_xprime_insertion_indices: int array of shape (n_samples)
        Insertion indices into X_knn_dists for points in Xprime.

    X_knn_mask: bool array of shape (n_samples, k)
        Masks out indices and distances from X_knn that are not valid. See note
        for AdapativeKNearestNeighbors.

    Xprime_to_X_dmat: float array of shape (n_samples, n_samples_2)
        Distances of points in X to points in Xprime.

    Xprime_to_X_knn_indices_rshifted: int array of shape (n_samples, k)
        Indices into the the pairwise distance matrix Xprime_to_X_dmat.
        Shifted by 1 to the right to speed up computation later.

    Xprime_to_X_knn_mask: bool array of shape (n_samples, k)
        Masks out indices and distances from Xprime_knn that are not valid. See
        note for AdapativeKNearestNeighbors.

    k: int
        The k for k-nearest neighbor search. Note that this definition of k
        will include the center point itself.

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
    P_Xprime = np.empty(Xprime_to_X_dmat.shape[0], dtype=np.float64)
    for i in numba.prange(Xprime_to_X_dmat.shape[0]):
        # Indices of xprime to points in **X**
        # Solution for xprime: This is a list of indices to the k-nearest neighbors of xprime
        xprime_knn_indices = Xprime_to_X_knn_indices_rshifted[i]
        xprime_knn_dists = Xprime_to_X_dmat[i]
        xprime_knn_mask = Xprime_to_X_knn_mask[i]

        # Indices of where xprime is located as a nearest neighbor of each Point in X
        xprime_insertion_indices = X_knn_dists_xprime_insertion_indices[i]

        # Insert the xprime (index 0) at the correct position into our sort indices
        # Solution for X: This is a 2-D array of indices to the k-nearest neighbors
        #                 for each point in X.
        X_knn_indices_with_xprime, X_knn_mask_with_xprime = insert_and_update_mask(
            X_knn_indices_rshifted,
            X_knn_mask,
            X_knn_dists,
            0.0,
            xprime_insertion_indices,
            xprime_knn_dists,
            k,
        )

        # Solution: knn_indices
        knn_indices = np.vstack(
            (np.expand_dims(xprime_knn_indices, 0), X_knn_indices_with_xprime)
        )

        # Solution: knn_mask
        knn_mask = np.vstack(
            (np.expand_dims(xprime_knn_mask, 0), X_knn_mask_with_xprime)
        )

        # We have to left shift the indices to use them to index into Xprime_to_X_dmat
        # We can do this inplace because we will not touch that row again
        np.subtract(xprime_knn_indices, 1, xprime_knn_indices)

        # Distance to itself is 0 due to prepend 0 to Xprime_to_X_dmat trick.
        # Solution xprime: This is a list of distances to the k-nearest neighbors of xprime
        xprime_knn_dists = xprime_knn_dists[xprime_knn_indices]

        # We have to left shift indices back to use them as indices into X_dmat.
        # We can do this inplace because we will not touch the indices again
        np.subtract(X_knn_indices_with_xprime, 1, X_knn_indices_with_xprime)
        # Get the distances from X_dmat
        X_knn_dists_with_xprime = parallel_take_along_axis(
            X_dmat, X_knn_indices_with_xprime
        )

        # Due to the left shift the 0 index of our xprime is now -1. We therefore determine
        # all indices where it is -1 and use them to select the correct distances from
        # Xprime_to_X_dmat instead of X_dmat
        new_point_indices = np.where(X_knn_indices_with_xprime == -1)
        # Get the distances from Xprime_to_X_dmat and override incorrect values in
        # X_knn_dists_with_xprime.
        # Solution X: This is a 2-D array of distances to the k-nearest neighbors for
        #             each point in X.
        # X_knn_dists_with_xprime[new_point_indices] = Xprime_to_X_dmat[i][new_point_indices[0]]
        parallel_put_by_advanced_index(
            X_knn_dists_with_xprime,
            new_point_indices,
            Xprime_to_X_dmat[i][new_point_indices[0]],
        )

        # Solution: knn_dists
        knn_dists = np.vstack(
            (np.expand_dims(xprime_knn_dists, 0), X_knn_dists_with_xprime)
        )

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
        # We also do not want xprime's edges (rows == 0, cols == 0)
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
        The k for k-nearest neighbor search. Note that this definition of k
        will include the center point itself.

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
        Impact is the fuzzy topology impact given X and Xprime.
    
    P_X: float
        Sum of edge probabilities in fuzzy simplicial sets of X.
    
    
    P_X_Xprime_minus_xprime: array of shape (n_samples_2)
        List of sum of edge probabilities of points in Xprime when added to X minus
        the edge probabilities of edges from and to points in Xprime.
    
    fs_set_X.size: int
        The number of edges in fuzzy simplicial sets of X.
    """
    X = X.astype(np.float64, copy=False)
    Xprime = Xprime.astype(np.float64, copy=False)

    # TODO Replace pairwise_distances with a faster numba version
    if X_dmat is None:
        X_dmat = pairwise_distances(X, metric=metric)

    X_knn_indices, X_knn_mask = parallel_adaptive_knn_indices(X_dmat, k)
    # TODO Copy might not be necessary
    X_knn_dists = X_dmat[np.arange(X_dmat.shape[0])[:, None], X_knn_indices].copy()
    X_knn_dists[np.logical_not(X_knn_mask)] = -1

    _, _, fs_set_X = fuzzy_simplicial_set(
        k, X_knn_indices, X_knn_dists, X_knn_mask, local_connectivity
    )

    P_X = np.sum(fs_set_X)

    # The "xprime" would be prependend
    # If train dmat = [5 x 5] it would be [6 x 6] with xprime prepended
    # We therefore shift the [5 x 5] indices by 1 to the right
    #  __         __
    # |0 _        _|
    # | |1 2 3 4 5 |
    # | |2         |
    # | |3         |
    # | |4         |
    # |_|5        _|
    #
    # Renaming is just for clarity
    X_knn_indices_rshifted = X_knn_indices
    np.add(X_knn_indices_rshifted, 1, out=X_knn_indices_rshifted)

    # TODO Replace pairwise_distances with a faster numba version
    # Row = xprime, Col = Distance to Train Point
    Xprime_to_X_dmat = pairwise_distances(Xprime, X, metric=metric)

    # Row = Point, Column = Where to insert
    X_knn_dists_xprime_insertion_indices = parallel_searchsorted(
        X_knn_dists, Xprime_to_X_dmat.T
    ).T

    # Row = xprime, Col = Index into dmat
    # Size = k (We are only interested in the k closest neighbors, including xprime itself)
    # Therefore we remove the furthest point (k - 1).
    # Xprime_to_X_dmat_argsort = parallel_argsort(Xprime_to_X_dmat)[:, : k - 1]
    Xprime_to_X_knn_indices, Xprime_to_X_knn_mask = parallel_adaptive_knn_indices(
        Xprime_to_X_dmat, k - 1
    )

    # Renaming is just for clarity
    Xprime_to_X_knn_indices_rshifted = Xprime_to_X_knn_indices
    # Shift by 1 due to prepend [see X_knn_indices_rshifted]
    np.add(Xprime_to_X_knn_indices_rshifted, 1, out=Xprime_to_X_knn_indices_rshifted)

    # Because our interpretation of k-NN includes the center itself we have to
    # prepend the center as the point with the closest distance (index 0)
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

    # X_dmat might be larger than Xprime_to_X_dmat due to identical farthest k-nearest neighbors
    Xprime_to_X_knn_indices_rshifted_tmp = np.zeros(
        (
            Xprime_to_X_knn_indices_rshifted.shape[0],
            X_knn_indices_rshifted.shape[1] + 1,
        ),
        dtype=Xprime_to_X_knn_indices_rshifted.dtype,
    )
    Xprime_to_X_knn_indices_rshifted_tmp[
        :, : Xprime_to_X_knn_indices_rshifted.shape[1]
    ] = Xprime_to_X_knn_indices_rshifted
    Xprime_to_X_knn_indices_rshifted = Xprime_to_X_knn_indices_rshifted_tmp

    # We therefore also have to prepend a True to our mask
    Xprime_to_X_knn_mask = np.concatenate(
        (
            np.ones(
                (Xprime_to_X_knn_mask.shape[0], 1), dtype=Xprime_to_X_knn_mask.dtype
            ),
            Xprime_to_X_knn_mask,
        ),
        axis=1,
    )

    # X_knn_mask might be larger than Xprime_to_X_knn_mask due to identical farthest k-nearest neighbors
    Xprime_to_X_mask_tmp = np.zeros(
        (Xprime_to_X_knn_mask.shape[0], X_knn_mask.shape[1] + 1),
        dtype=Xprime_to_X_knn_mask.dtype,
    )
    Xprime_to_X_mask_tmp[:, : Xprime_to_X_knn_mask.shape[1]] = Xprime_to_X_knn_mask
    Xprime_to_X_knn_mask = Xprime_to_X_mask_tmp

    # We append 0 distances to later map the knn_index of -1 to a distance of 0
    Xprime_to_X_dmat = np.concatenate(
        (
            Xprime_to_X_dmat,
            np.zeros((Xprime_to_X_dmat.shape[0], 1), dtype=Xprime_to_X_dmat.dtype),
        ),
        axis=1,
    )

    P_X_Xprime_minus_xprime = fti(
        X_dmat,
        X_knn_dists,
        X_knn_indices_rshifted,
        X_knn_dists_xprime_insertion_indices,
        X_knn_mask,
        Xprime_to_X_dmat,
        Xprime_to_X_knn_indices_rshifted,
        Xprime_to_X_knn_mask,
        k,
        local_connectivity,
    )
    impact = (P_X - np.mean(P_X_Xprime_minus_xprime)) / fs_set_X.size
    return impact, P_X, P_X_Xprime_minus_xprime, fs_set_X.size
