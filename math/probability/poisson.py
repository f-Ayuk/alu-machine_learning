#!/usr/bin/env python3
"""creating a class that represents a poisson distribution."""


class Poisson:
    """Creating the Poisson class
    """

    def __init__(self, data=None, lambtha=1.):
        """ Constructor
        """
        self.E = 2.7182818285
        self.PI = 3.1415926536

        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates the value of the PMF (probability mass function)
        for a given number of “successes”"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        p = (self.lambtha ** k) * (self.E ** -self.lambtha) / self.factorial(k)
        return p
