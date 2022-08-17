import scipy.stats as stats

def generateRandomSample(mu, std, count):
    assert mu is not None, 'Mean of the distribution is missing'
    assert std is not None, 'Std Dev of the distribution is missing'
    assert count is not None, 'Number of values to return is missing'
    return stats.norm.rvs(loc=mu,scale=std,size=count)