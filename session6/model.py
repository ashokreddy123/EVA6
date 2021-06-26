
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


drop_out = 0.05



class Net_batch(nn.Module):

  def __init__(self):
    super(Net_batch, self).__init__()
    
    #conv block
    self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(drop_out)
    ) #input_size = 28, #output_size = 26, RF = 3
    self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop_out)
    ) #input_size = 26, #output_size = 24, RF = 5
    """
    self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(drop_out)
    ) #output_size = 22
   """
    # transition block
    self.maxpool1 = nn.MaxPool2d(2,2) #input_size = 24, #output_size = 12, RF = 6
    self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(drop_out)
    )#input_size = 12, #output_size = 10, RF = 10

    #convolutuonla block 2
    self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(drop_out)
    )#input_size = 10, #output_size = 8, RF = 14
    self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop_out)
    )#input_size = 8, #output_size = 6, RF = 18
    self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop_out)
    )#input_size = 6, #output_size = 4, RF = 22
    self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop_out)
    )#input_size = 4, #output_size = 2, RF = 26

    #output block
    self.gap = nn.AdaptiveAvgPool2d((1,1)) #input_size = 2, #output_size = 1, RF = 28
    self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            
    )

  def forward(self,input):

    #x = self.convblock3(self.convblock2(self.convblock1(input)))
    x = self.convblock2(self.convblock1(input))

    x = self.convblock4(self.maxpool1(x))

    x = self.convblock8(self.convblock7(self.convblock6(self.convblock5(x))))

    x = self.convblock9(self.gap(x))

    return F.log_softmax(x)



class Net_layer(nn.Module):

  def __init__(self):
    super(Net_layer, self).__init__()
    
    #conv block
    self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            #nn.BatchNorm2d(8),
            nn.GroupNorm(1, 8),
            nn.ReLU(),
            nn.Dropout(drop_out)
    ) #input_size = 28, #output_size = 26, RF = 3
    self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(1, 16),
            nn.ReLU(),
            nn.Dropout(drop_out)
    ) #input_size = 26, #output_size = 24, RF = 5
    """
    self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(drop_out)
    ) #output_size = 22
   """
    # transition block
    self.maxpool1 = nn.MaxPool2d(2,2) #input_size = 24, #output_size = 12, RF = 6
    self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.GroupNorm(1, 8),
            nn.ReLU(),
            nn.Dropout(drop_out)
    )#input_size = 12, #output_size = 10, RF = 10

    #convolutuonla block 2
    self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(1, 8),
            nn.ReLU(),
            nn.Dropout(drop_out)
    )#input_size = 10, #output_size = 8, RF = 14
    self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(1, 16),
            nn.ReLU(),
            nn.Dropout(drop_out)
    )#input_size = 8, #output_size = 6, RF = 18
    self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(1, 16),
            nn.ReLU(),
            nn.Dropout(drop_out)
    )#input_size = 6, #output_size = 4, RF = 22
    self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(1, 16),
            nn.ReLU(),
            nn.Dropout(drop_out)
    )#input_size = 4, #output_size = 2, RF = 26

    #output block
    self.gap = nn.AdaptiveAvgPool2d((1,1)) #input_size = 2, #output_size = 1, RF = 28
    self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            
    )

  def forward(self,input):

    #x = self.convblock3(self.convblock2(self.convblock1(input)))
    x = self.convblock2(self.convblock1(input))

    x = self.convblock4(self.maxpool1(x))

    x = self.convblock8(self.convblock7(self.convblock6(self.convblock5(x))))

    x = self.convblock9(self.gap(x))

    return F.log_softmax(x)



class Net_group(nn.Module):

  def __init__(self):
    super(Net_group, self).__init__()
    
    #conv block
    self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            #nn.BatchNorm2d(8),
            nn.GroupNorm(4, 8),
            nn.ReLU(),
            nn.Dropout(drop_out)
    ) #input_size = 28, #output_size = 26, RF = 3
    self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Dropout(drop_out)
    ) #input_size = 26, #output_size = 24, RF = 5
    """
    self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(drop_out)
    ) #output_size = 22
   """
    # transition block
    self.maxpool1 = nn.MaxPool2d(2,2) #input_size = 24, #output_size = 12, RF = 6
    self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.GroupNorm(4, 8),
            nn.ReLU(),
            nn.Dropout(drop_out)
    )#input_size = 12, #output_size = 10, RF = 10

    #convolutuonla block 2
    self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(4, 8),
            nn.ReLU(),
            nn.Dropout(drop_out)
    )#input_size = 10, #output_size = 8, RF = 14
    self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Dropout(drop_out)
    )#input_size = 8, #output_size = 6, RF = 18
    self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Dropout(drop_out)
    )#input_size = 6, #output_size = 4, RF = 22
    self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Dropout(drop_out)
    )#input_size = 4, #output_size = 2, RF = 26

    #output block
    self.gap = nn.AdaptiveAvgPool2d((1,1)) #input_size = 2, #output_size = 1, RF = 28
    self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            
    )

  def forward(self,input):

    #x = self.convblock3(self.convblock2(self.convblock1(input)))
    x = self.convblock2(self.convblock1(input))

    x = self.convblock4(self.maxpool1(x))

    x = self.convblock8(self.convblock7(self.convblock6(self.convblock5(x))))

    x = self.convblock9(self.gap(x))

    return F.log_softmax(x)

def network_model(norm_arg):

	
	if(norm_arg == 'batch'):
		print("Batch is selected")
		model = Net_batch()
	if(norm_arg == 'layer'):
		print("Layer is selected")
		model = Net_layer()
	if(norm_arg == 'group'):
		print("Group is selected")
		model = Net_group()

	return model



