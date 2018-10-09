from .dinknet import *

class DlinkNetLate(DinkNet34):
    def __init__(self):
        super().__init__(num_classes=7, num_channels=3)
        
    def forward(self, x_stacked):
        x1 = x_stacked[:3]
        x2 = x_stacked[3:]
        
        