import torch
import torch.nn as nn
import torch.nn.functional as F

class GradReverse(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) help learn adversarial features by reversing the sign of the gradient
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):     # Reverse the sign of the input gradient and multiply it by constant
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class fea_Extractor_global(nn.Module):
    """
    Extract global features from the input data and return the features
    """
    def __init__(self, in_channels, out_channels):
        super(fea_Extractor_global, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding='same')
        self.dropout1 = nn.Dropout()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, padding='same')
        self.dropout2 = nn.Dropout()
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=out_channels, kernel_size=2, padding='same')
        self.dropout3 = nn.Dropout()

    def forward(self, input):
        x = self.conv1(input)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = torch.relu(self.conv3(x))
        x = self.dropout3(x)
        return x

class fea_Extractor_att1(nn.Module):
    """
    Extract the first-layer attribute features from the input data and return the features
    """
    def __init__(self, in_channels, out_channels):
        super(fea_Extractor_att1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=20, kernel_size=3, stride=1, padding='same')
        self.dropout1 = nn.Dropout()
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=out_channels, kernel_size=2, padding='same')
        self.dropout2 = nn.Dropout()

    def forward(self, input):
        x = torch.relu(self.conv1(input))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        return x

class fea_Extractor_att2(nn.Module):
    """
    Extract the second-layer attribute features from the input data and return the features
    """
    def __init__(self, in_channels, out_channels):
        super(fea_Extractor_att2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=20, kernel_size=3, stride=1, padding='same')
        self.dropout1 = nn.Dropout()
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=out_channels, kernel_size=2, padding='same')
        self.dropout2 = nn.Dropout()

    def forward(self, input):
        x = torch.relu(self.conv1(input))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        return x

class fea_Extractor_att3(nn.Module):
    """
    Extract the third-layer attribute features from the input data and return the features
    """
    def __init__(self, in_channels, out_channels):
        super(fea_Extractor_att3, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=20, kernel_size=3, stride=1, padding='same')
        self.dropout1 = nn.Dropout()
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=out_channels, kernel_size=2, padding='same')
        self.dropout2 = nn.Dropout()

    def forward(self, input):
        x = torch.relu(self.conv1(input))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        return x

class Att_transmission2(nn.Module):
    """
    ACA2
    """
    def __init__(self):
        super(Att_transmission2, self).__init__()
        self.fc1 = nn.Linear(70, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 70)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Att_transmission3(nn.Module):
    """
    ACA3
    """
    def __init__(self):
        super(Att_transmission3, self).__init__()
        self.fc1 = nn.Linear(70, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 70)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Att_predictor1(nn.Module):
    """
    The first-layer attribute predictor
    """
    def __init__(self):
        super(Att_predictor1, self).__init__()
        self.fc1 = nn.Linear(70, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 3)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.fc2(F.dropout(x))
        x = F.relu(x)
        x = self.fc3(x)
        return x

class Att_predictor2(nn.Module):
    """
    The second-layer attribute predictor
    """
    def __init__(self):
        super(Att_predictor2, self).__init__()
        self.fc1 = nn.Linear(70, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.fc2(F.dropout(x))
        x = F.relu(x)
        x = self.fc3(x)
        return x

class Att_predictor3(nn.Module):
    """
    The third-layer attribute predictor
    """
    def __init__(self):
        super(Att_predictor3, self).__init__()
        self.fc1 = nn.Linear(70, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 3)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.fc2(F.dropout(x))
        x = F.relu(x)
        x = self.fc3(x)
        return x

class Domain_classifier_att1(nn.Module):
    """
    The first-layer domain discriminators, return the logits after log_softmax
    """
    def __init__(self):
        super(Domain_classifier_att1, self).__init__()
        self.fc1 = nn.Linear(70, 30)
        self.fc2 = nn.Linear(30, 2)

    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)   # GRL
        logits = F.relu(self.fc1(input))
        logits = F.log_softmax(self.fc2(logits), 1)
        return logits

class Domain_classifier_att2(nn.Module):
    def __init__(self):
        super(Domain_classifier_att2, self).__init__()
        self.fc1 = nn.Linear(70, 30)
        self.fc2 = nn.Linear(30, 2)

    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)   # GRL
        logits = F.relu(self.fc1(input))
        logits = F.log_softmax(self.fc2(logits), 1)
        return logits

class Domain_classifier_att3(nn.Module):
    def __init__(self):
        super(Domain_classifier_att3, self).__init__()
        self.fc1 = nn.Linear(70, 30)
        self.fc2 = nn.Linear(30, 2)

    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)   # GRL
        logits = F.relu(self.fc1(input))
        logits = F.log_softmax(self.fc2(logits), 1)
        return logits

class Classifier(nn.Module):
    """
    State classifier, return the original logits including both known and unknown classes
    """
    def __init__(self, output_size=6):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(64 * 7, 20)
        self.fc2 = nn.Linear(8, 20)
        self.fc3 = nn.Linear(20 * 2, output_size)

    def forward(self, input_1, input_2, constant=1, adaption=False):
        input_1 = input_1.view(input_1.size(0), -1)
        if adaption == True:
            input_1 = GradReverse.grad_reverse(input_1, constant)   # GRL
        x1 = F.relu(self.fc1(input_1))
        x2 = F.relu(self.fc2(input_2))
        x = torch.cat((x1, x2), dim=1)
        x = self.fc3(F.dropout(x))
        return x