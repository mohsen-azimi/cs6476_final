
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        #      , , ]
        layers = []
        # 64, 64, 'M'
        layers += [nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1),
                   nn.BatchNorm2d(64),
                   nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
                   nn.BatchNorm2d(64),
                   nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Dropout(0.2)]



        # 128, 128, 'M'
        layers += [nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
                   nn.BatchNorm2d(128),
                   nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
                   nn.BatchNorm2d(128),
                   nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Dropout(0.2)]

       #  256, 256, 256, 256, 'M'
        layers += [nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
                   nn.BatchNorm2d(256),
                   nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
                   nn.BatchNorm2d(256),
                   nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
                   nn.BatchNorm2d(256),
                   nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
                   nn.BatchNorm2d(256),
                   nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Dropout(0.2)]

        # 512, 512, 512, 512, 'M'
        layers += [nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
                   nn.BatchNorm2d(512),
                   nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
                   nn.BatchNorm2d(512),
                   nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
                   nn.BatchNorm2d(512),
                   nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
                   nn.BatchNorm2d(512),
                   nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Dropout(0.2)]

        # 512, 512, 512, 512, 'M'
        layers += [nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
                   nn.BatchNorm2d(512),
                   nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
                   nn.BatchNorm2d(512),
                   nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
                   nn.BatchNorm2d(512),
                   nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
                   nn.BatchNorm2d(512),
                   nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Dropout(0.2)]


        # average pooling
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]


        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 11),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




if __name__ == '__main__':
    model = MyModel()
    inputs = torch.randn(5, 3, 32, 32)
    output = model(inputs)

    print(output.size())



