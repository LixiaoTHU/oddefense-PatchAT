import random
from ..builder import PIPELINES


@PIPELINES.register_module()
class Clipcheck:

    def __init__(self, pmin=0, pmax=255):
        self.pmin = pmin
        self.pmax = pmax


    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key] 
            img[img < self.pmin] = self.pmin
            img[img > self.pmax] = self.pmax
        return results