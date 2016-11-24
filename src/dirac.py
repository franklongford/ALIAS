class DiracDelta:
    """
    Smoothed Dirac delta function:
    $\frac{1}{2\epsilon}(1 + \cos(\pi x/\epsilon)$ when
    $x\in [-\epsilon, \epsilon]$ and 0 elsewhere.
    """

    def __init__(self, eps, vectorized=False):
        self.eps = eps
        if self.eps == 0:
            raise ValueError('eps=0 is not allowed in class DiracDelta.')


    def __call__(self, x):
        if isinstance(x, (float, int)):
            return _smooth(x)
        elif isinstance(x, ndarray):
            return _smooth_vec(x)
        else:
            raise TypeError('%s x is wrong' % type(x))

    def _smooth(self, x):
        eps = self.eps
        if x < -eps or x > eps:
            return 0
        else:
            return 1./(2*eps)*(1 + cos(pi*x/eps))

    def _smooth_vec(self, x):
        eps = self.eps
        r = zeros_like(x)
        condition1 - operator.and_(x >= -eps, x <= eps)
        xc = x[condition1]
        r[condition1] = 1./(2*eps)*(1 + cos(pi*xc/eps))
        return r


    def plot(self, center=0, xmin=-1, xmax=1):
        """
        Return arrays x, y for plotting the DiracDelta function
        centered in `center` on the interval [`xmin`, `xmax`].
        """
        n = 200./self.eps
        x = concatenate(
            linspace(xmin, center-self.eps, 21),
            linspace(center-self.eps, center+self.eps, n+1),
            linspace(center+self.eps, xmax, 21))
        y = self(x)
        return x, y

