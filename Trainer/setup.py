from distutils.core import setup
#files = ["things/*"]

setup(name='DLEngine',
      version='0.0.1',
      packages=['DLEngine',
                'DLEngine.modules',
                'DLEngine.modules.dataloader',
                'DLEngine.modules.dataloader.dataset',
                'DLEngine.modules.lr_schedule',
                'DLEngine.modules.metric',
                'DLEngine.modules.optimizer',
                'DLEngine.modules.trainer']
     )

