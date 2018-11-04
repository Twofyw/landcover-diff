from .dlinknet_original import *

cuts = {
        'DLinkNet34': [8, 8],
        }

def get(arch, **kwargs):
    if arch == 'DLinkNet34':
        num_classes = kwargs.get('num_classes', None)
        run_lastlayer = kwargs.get('run_lastlayer', None)
        return DLinkNet34(num_classes=num_classes,
                run_lastlayer=run_lastlayer), cuts[arch]
    else:
        raise NotImplementedError

