import numpy as np
import scipy.stats as stats


class DataSimulator:
    """Class for simulating data for a power analysis.

    Parameters
    ----------
    paramlist : list of lists of parameters
        See notes.
    random_state : int or numpy.random.Generator instance, optional
        Seedable for reproducibility, by default None

    Examples
    --------
    Each sublist in paramlist can either be an integer or a scipy.stats
    random distribution generator. The following lines illustrate a few
    ways of specifying the same parameters (no treatment effect, both
    columns are randomly generated Gaussian variables).

    >>> paramlist = [[0, 0], [[stats.norm]]*6, [stats.norm, 0, 1]]
    >>> paramlist = [[0]*2, [stats.norm], [stats.norm]]

    """

    def __init__(self, paramlist, random_state=None):

        self.parameters = paramlist
        self.random_generator = np.random.default_rng(random_state)

    def fit(self, hierarchy=[]):
        """Fit the DataSimulator to a hierarchy.

        Parameters
        ----------
        hierarchy : list of ints, optional
            number of clusters in each column, by default []

        Examples
        --------
        This creates a data container with 2 clusters in column 0, 3 clusters each
        in column 1 (for 6 total), and 3 clusters each in column 2 (18 total).

        >>> import scipy.stats as stats
        >>> paramlist = [[0, 0], [[stats.norm]]*8, [stats.norm, 0, 1]]
        >>> datagen = DataSimulator(paramlist)
        >>> hierarchy = [2, 3, 3]
        >>> datagen.fit(hierarchy)
        >>> datagen.container
        array([[1., 1., 1., 0.],
               [1., 1., 2., 0.],
               [1., 1., 3., 0.],
               [1., 2., 1., 0.],
               [1., 2., 2., 0.],
               [1., 2., 3., 0.],
               [1., 3., 1., 0.],
               [1., 3., 2., 0.],
               [1., 3., 3., 0.],
               [2., 1., 1., 0.],
               [2., 1., 2., 0.],
               [2., 1., 3., 0.],
               [2., 2., 1., 0.],
               [2., 2., 2., 0.],
               [2., 2., 3., 0.],
               [2., 3., 1., 0.],
               [2., 3., 2., 0.],
               [2., 3., 3., 0.]])
        

        """
        self.container = _make_ref_container(hierarchy.copy())

    def generate(self):
        """Generate a simulated dataset based on specified parameters and hierarchy.

        Returns
        -------
        2D numeric
            Simulated data.

        Examples
        --------
        >>> paramlist = [[0, 0], [stats.norm], [stats.norm]]
        >>> datagen = DataSimulator(paramlist, random_state=1)
        >>> hierarchy = [2, 3, 3]
        >>> datagen.fit(hierarchy)
        >>> datagen.generate()
        array([[ 1.        ,  1.        ,  1.        , -0.19136904],
               [ 1.        ,  1.        ,  2.        ,  0.9267023 ],
               [ 1.        ,  1.        ,  3.        ,  0.71015659],
               [ 1.        ,  2.        ,  1.        ,  1.11575064],
               [ 1.        ,  2.        ,  2.        ,  0.85004038],
               [ 1.        ,  2.        ,  3.        ,  1.36833113],
               [ 1.        ,  3.        ,  1.        , -0.40601701],
               [ 1.        ,  3.        ,  2.        ,  0.16752713],
               [ 1.        ,  3.        ,  3.        , -0.15168224],
               [ 2.        ,  1.        ,  1.        , -0.70431102],
               [ 2.        ,  1.        ,  2.        , -1.26343512],
               [ 2.        ,  1.        ,  3.        , -1.59561398],
               [ 2.        ,  2.        ,  1.        ,  0.1234474 ],
               [ 2.        ,  2.        ,  2.        ,  0.64816363],
               [ 2.        ,  2.        ,  3.        ,  0.91349805],
               [ 2.        ,  3.        ,  1.        ,  0.17077167],
               [ 2.        ,  3.        ,  2.        ,  1.74043839],
               [ 2.        ,  3.        ,  3.        ,  1.45309889]])
        """
        output = self.container.copy()
        output[:, -1] = _gen_fake_data(
            self.container, self.parameters, random_state=self.random_generator
        )
        return output


def _gen_fake_data(reference, params, random_state=None):
    """Generates y-values for a design matrix.

    Parameters
    ----------
    reference : 2D array
        Design matrix used to simulate data.
    params : list of lists
        Parameters specifying the distributions of the simulated data.
        See DataSimulator for details.
    random_state : int or numpy.random.Generator instance, optional
        Seedable for reproducibility, by default None

    Returns
    -------
    1D array of floats
        Simulated y-values.
    """
    rng = np.random.default_rng(random_state)

    fakedata = np.copy(reference)
    fakedata = fakedata.astype("float64")
    for i in range(reference.shape[1] - 1):

        if type(params[i][0]) is not int:

            idx, replicates = np.unique(
                reference[:, : i + 1], return_counts=True, axis=0
            )
            ranlist = []
            if not isinstance(params[i][0], list):
                for j in range(len(idx)):
                    ranlist.append(params[i][0].rvs(*params[i][1:], random_state=rng))
                ranlist = np.repeat(ranlist, replicates)
            else:
                for j in range(len(idx)):
                    ranlist.append(
                        params[i][j][0].rvs(*params[i][j][1:], random_state=rng)
                    )
                ranlist = np.repeat(ranlist, replicates)

            np.put(fakedata[:, i], np.where(fakedata[:, i]), ranlist)

        else:
            idx, replicates = np.unique(
                reference[:, : i + 1], return_counts=True, axis=0
            )
            ranlist = np.repeat(params[i], replicates)
            np.put(fakedata[:, i], np.where(fakedata[:, i] > -1), ranlist)

        if i > 0:
            fakedata[:, i] = fakedata[:, i] + fakedata[:, i - 1]
    else:
        return fakedata[:, -2]


def _make_ref_container(samples_per_level=[]):
    """Converts a hierarchy list into a design matrix.

    Parameters
    ----------
    samples_per_level : list, optional
        List of ints specifying hierarchy, by default []

    Returns
    -------
    2D array of floats
        Design matrix specified by samples_per_level
    """
    iterator = list(samples_per_level)

    hcontainer = np.arange(1, iterator[0] + 1).reshape(iterator[0], 1)

    for i in range(1, len(iterator)):
        if type(iterator[i]) is not list:
            iterator[i] = [iterator[i]] * hcontainer.shape[0]
        hcontainer = np.repeat(hcontainer, iterator[i], axis=0)
        append = []
        for j in iterator[i]:
            append += list(range(1, j + 1))
        hcontainer = np.append(
            hcontainer, np.array(append).reshape(len(append), 1), axis=1
        )
    hcontainer = np.append(
        hcontainer,
        np.zeros_like(hcontainer[:, 0]).reshape(hcontainer[:, 0].shape[0], 1),
        axis=1,
    )

    return hcontainer.astype(float)
