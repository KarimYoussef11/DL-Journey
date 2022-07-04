import torch


class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1=torch.randn(linear_1_out_features, linear_1_in_features), # 20 x 2
            b1=torch.randn(linear_1_out_features),                       # 20
            W2=torch.randn(linear_2_out_features, linear_2_in_features), # 5 x 20
            b2=torch.randn(linear_2_out_features),                       # 5
        )
        self.grads = dict(
            dJdW1=torch.zeros(linear_1_out_features, linear_1_in_features), # 20 x 2
            dJdb1=torch.zeros(linear_1_out_features),                       # 20
            dJdW2=torch.zeros(linear_2_out_features, linear_2_in_features), # 5 x 20
            dJdb2=torch.zeros(linear_2_out_features),                       # 1 x 5
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        self.z1 = torch.matmul(x, self.parameters['W1'].T) + self.parameters['b1']
        self.z2 = self.nonlinear(self.f_function, self.z1)
        self.z3 = torch.matmul(self.z2, self.parameters['W2'].T) + self.parameters['b2']
        self.x = x
        y_hat = self.nonlinear(self.g_function, self.z3)
        return y_hat

    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        dJdz3 = dJdy_hat
        dz3db2 = torch.ones(dJdy_hat.shape[0])
        dz2dz1 = torch.zeros(self.z2.shape[0], self.z2.shape[1])
        dz2dz1[self.z2 > 0] = 1
        dz1db1 = torch.ones(dJdy_hat.shape[0])
        self.grads['dJdW2'] = torch.matmul(dJdz3.T, self.z2)
        self.grads['dJdb2'] = torch.matmul(dz3db2, dJdz3)
        self.grads['dJdW1'] = torch.matmul((torch.matmul(dJdz3, self.parameters['W2']) * dz2dz1).T, self.x)
        self.grads['dJdb1'] = torch.matmul((torch.matmul(dJdz3, self.parameters['W2']) * dz2dz1).T, dz1db1)

    def nonlinear(self, fun, zi):
        if fun == 'relu':
            zip = torch.relu(zi)
        elif fun == 'sigmoid':
            zip = torch.sigmoid(zi)
        else:
            zip = zi
        return zip

    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()


def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    loss = 1/y.numel() * torch.sum((y_hat - y) ** 2)
    dJdy_hat = 2/y.numel() * (y_hat - y)                          # 10 x 5
    return loss, dJdy_hat


def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    loss = torch.sum(y * torch.log(y_hat) + (1-y) * torch.log(1-y_hat))
    dJdy_hat = -y/y_hat + (1-y) / (1-y_hat)
    return loss, dJdy_hat

# Refer to the word document for better explanation on how the gradients were calculated for back prop.
