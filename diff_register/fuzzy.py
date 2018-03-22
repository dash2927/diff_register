import numpy as np


def triangmf(z, a, b, c):
    """

    Computes a fuzzy membership function with a triangular shape.

    Adapted from Digital Image Processing Using MATLAB (2nd Ed.) by R.C.
    Gonzalez, R. E. Woods, and S. L. Eddins.

    Inputs
    ------
    z: input variable, can be a vector of any length.
    a: scalar parameter, such that b >= a
    b: scalar parameter, such that b >= a, c >= b
    c: scalar parameter, such that c >= b

    Returns
    -------
    mu: fuzzy membership function

    Examples
    --------

    """

    mu = np.zeros(z.shape)
    low_side = (a <= z) & (z < b)
    high_side = (b <= z) & (z < c)
    mu[low_side] = (z[low_side] - a) / (b - a)
    mu[high_side] = 1 - (z[high_side] - b) / (c - b)

    return mu


def trapezmf(z, a, b, c, d):
    """

    Computes a fuzzy membership function with a trapezoidal shape.

    Adapted from Digital Image Processing Using MATLAB (2nd Ed.) by R.C.
    Gonzalez, R. E. Woods, and S. L. Eddins.

    Inputs
    ------
    z: input variable, can be a vector of any length.
    a: scalar parameter, such that b >= a
    b: scalar parameter, such that b >= a, c >= b
    c: scalar parameter, such that c <= d
    d: scalar parameter

    Returns
    -------
    mu: fuzzy membership function

    Examples
    --------

    """

    mu = np.zeros(z.shape)
    up_ramp_region = (a <= z) & (z < b)
    top_region = (b <= z) & (z < c)
    down_ramp_region = (c <= z) & (z < d)

    mu[up_ramp_region] = 1 - (b - z[up_ramp_region]) / (b - a)
    mu[top_region] = 1
    mu[down_ramp_region] = 1 - (z[down_ramp_region] - c) / (d - c)

    return mu


def sigmamf(z, a, b):
    """

    Computes the sigma fuzzy membership function.

    Adapted from Digital Image Processing Using MATLAB (2nd Ed.) by R.C.
    Gonzalez, R. E. Woods, and S. L. Eddins.

    Inputs
    ------
    z: input variable, can be a vector of any length.
    a: scalar parameter, such that b >= a
    b: scalar parameter, such that b >= a, c >= b

    Returns
    -------
    mu: fuzzy membership function

    Examples
    --------

    """

    mu = trapezmf(z, a, b, np.inf, np.inf)

    return mu


def smf(z, a, b):
    """

    Computes an s-shaped fuzzy membership function.

    Adapted from Digital Image Processing Using MATLAB (2nd Ed.) by R.C.
    Gonzalez, R. E. Woods, and S. L. Eddins.

    Inputs
    ------
    z: input variable, can be a vector of any length.
    a: scalar parameter, such that b >= a
    b: scalar parameter, such that b >= a, c >= b

    Returns
    -------
    mu: fuzzy membership function

    Examples
    --------

    """

    mu = np.zeros(z.shape)

    p = (a + b)/2
    low_range = (z <= z) & (z < p)
    mu[low_range] = 2 * ((z[low_range] - a) / (b - a))**2
    mid_range = (p <= z) & (z < b)
    mu[mid_range] = 1 - 2 * ((z[mid_range] - b) / (b - a))**2
    high_range = (b <= z)
    mu[high_range] = 1

    return mu


def bellmf(z, a, b):
    """

    Computes the bell-shaped fuzzy membership function.

    Adapted from Digital Image Processing Using MATLAB (2nd Ed.) by R.C.
    Gonzalez, R. E. Woods, and S. L. Eddins.

    Inputs
    ------
    z: input variable, can be a vector of any length.
    a: scalar parameter, such that b >= a
    b: scalar parameter, such that b >= a

    Returns
    -------
    mu: fuzzy membership function

    Examples
    --------

    """

    mu = np.zeros(z.shape)

    left_side = z < b
    mu[left_side] = smf(z[left_side], a, b)
    right_side = z >= b
    mu[right_side] = smf(2*b - z[right_side], a, b)

    return mu


def truncgaussmf(z, a, b, s):
    """

    Computes a truncated Gaussian fuzzy membership function.

    Adapted from Digital Image Processing Using MATLAB (2nd Ed.) by R.C.
    Gonzalez, R. E. Woods, and S. L. Eddins.

    Inputs
    ------
    z: input variable, can be a vector of any length.
    a: scalar parameter, such that b >= a
    b: scalar parameter, such that b >= a

    Returns
    -------
    mu: fuzzy membership function

    Examples
    --------

    """

    mu = np.zeros(z.shape)

    c = a + 2 * (b - a)
    range = (a <= z) & (z <= c)
    mu[range] = np.exp(-(z[range] - b)**2 / s**2)

    return mu


def zeromf(z):
    """

    Compus a zero memberoship function.

    Adapted from Digital Image Processing Using MATLAB (2nd Ed.) by R.C.
    Gonzalez, R. E. Woods, and S. L. Eddins.

    Inputs
    ------
    z: input variable, can be a vector of any length.

    Returns
    -------
    mu: fuzzy membership function

    Examples
    --------

    """

    mu = np.zeros(z.shape)

    return mu


def onemf(z):
    """

    Compus a one memberoship function.

    Adapted from Digital Image Processing Using MATLAB (2nd Ed.) by R.C.
    Gonzalez, R. E. Woods, and S. L. Eddins.

    Inputs
    ------
    z: input variable, can be a vector of any length.

    Returns
    -------
    mu: fuzzy membership function

    Examples
    --------

    """

    mu = np.ones(z.shape)

    return mu


def lambdafcns(inmf, op=lambda x: min(x)):
    """
    Creates a set of lambda functions (rule strength functions) corresponding to
    a set of fuzzy rules.

    Inputs
    ------
    inmf: M by N matrix of input membership function handle, where M is the
        number of rules, and N is the number of fuzzy input systems. inmf(i, j)
        is the input membership function applied by the i-th rule to the j-th
        input.

    op: a function handle used to combine the antecedents for each rule.  op can
        be either

    Returns
    -------
    L: array of function handles

    Examples
    --------

    """
    def listoflists(row, col):
        y = [[0]*col]*row
        return y

    def rulestrength(i, args):
        counter = 0
        for num in args:
            if counter == 0:
                Z = num

            # initialize lambda as the output of the first memberoship function of
            # the k-th rule.
            memberfcn = inmf[i][0]
            lda = memberfcn(Z[0])
            if counter > 0:
                memberfcn = inmf[i][counter]
                lda = op(lda, memberfcn(Z[counter]))

    num_rules = inmf.shape[0]
    L = listoflists(1, num_rules)

    for i in range(0, num_rules):
        # Each output lambda function calls the rulestrength() function with i
        # (to identify which row of the rules matrix should be used), followed
        # by all the z input arguments (which are passed along via *args)

        L[i] = lambda x: rulestrength(i, x)

    return L


def implfcns(L, outmf, args):
    """
    Creates a set of implication functions from a set of lambda functions L, a
    set of output member functions outmf, and a set of fuzzy system inputs
    args = [Z1, Z2, ..., ZN].

    Inputs
    ------
    L: a list of lists of rule-strength function handles as returned by
        lambdafcns
    outmf: a list of lists of output membership functions.  The number of
        elements of outmf can be either...

    Returns
    -------
    Q: a list of lists of implication function handles


    """

    def listoflists(row, col):
        y = [[0]*col]*row
        return y

    def implication(i, v):
        q = min(lambdas[i], outmf[i](v))
        return q

    def elserule(v):
        lambda_e = min(1 - lambdas)
        q = min(lambda_e, outmf[-1](v))
        return q

    Z = args[0]
    num_rules = len(L)
    Q = listoflists(len(outmf), 1)
    lambdas = listoflists(1, num_rules)

    for i in range(0, num_rules):
        # Each output implication function calls implication() with i to
        # identify which lambda value should be used, followed by v.

        Q[i] = lambda v: implication(i, v)

    if len(outmf) == num_rules + 1:
        Q[num_rules + 1] = lambda x: elserule(x)

    return Q


def aggfcn(Q):
    """
    Creates an aggregation function Qa from a set of implication functions Q.

    Inputs
    ------
    Q: list of lists of function handles as returned by implfcns

    Returns
    -------
    Qa: a function handle that can be called with a single input V.

    Examples
    --------

    """

    def aggregate(v):
        q = Q[0](v)
        for i in range(1, len(Q)):
            q = max(q, Q[i](v))

    Qa = lambda x: aggregate(x)

    return Qa


def defuzzify(Qa, vrange):
    """
    Transforms the aggregation function Qa into a fuzzy result using the
    center-of-gravity method.

    Inputs
    ------
    Qa: a function handle as returned by aggfcn.
    vrange: two-element vector specifying the range of input values for Qa.

    Returns
    -------
    out: scalar result
    """

    v1 = vrange[0]
    v2 = vrange[1]

    v = np.linspace(v1, v2, 100)
    Qv = Qa(v)
    out = np.sum(v*Qv) / np.sum(Qv)
    if np.isnan(out):
        # If Qv is zero everywhere, out will be NaN.  Arbitrarily choose output
        # to be the midpoint of vrange.
        out = np.mean(vrange)

    return out


def fuzzysysfcn(inmf, outmf, vrange, op=lambda x: min(x)):
    """
    Creates a fuzzy system function F, corresponding to a set of rules and
    output membership functions.

    Inputs
    ------
    inmf: M by N list of lists, M is the number of rules, N is the number of
        fuzzy system inputs.
    outmf: list of lists containing output membership functions. len(outmf) can
        be either M or M+1.
    vrange: two-element vector specifying the valid range of input values for
        the output membership functions.
    op: a function handle specifying how to combine the antecedents for each
        rule.  Can be either min or max.

    Returns
    -------
    F: a function handle that computes the fuzzy system's output, given a set of
        inputs, using the syntax...


    """
    F = 1

    return F
