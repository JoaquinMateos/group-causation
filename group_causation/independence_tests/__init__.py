from .hsic import HSIC_Test
from .max_corr import MaxCorr_Test



conditional_independence_tests = {
    'hsic': HSIC_Test,
    'max_corr': MaxCorr_Test,
}