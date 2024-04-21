if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
import matplotlib.pyplot as plt

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

x, t = dezero.datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)
data_size = len(x)
max_iter = math.ceil(data_size / batch_size)
loss_list = []

for epoch in range(max_epoch):
    idx = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_idx = idx[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_idx]
        batch_t = t[batch_idx]

        y = model(batch_x)
        loss = F.softmax_cross_entropy_simple(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    loss_list.append(avg_loss)
    if epoch % 10 == 0:
        print(f"epoch {epoch+1}, loss {avg_loss:.2f}")
        
plt.plot(range(max_epoch), loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')

plt.show()