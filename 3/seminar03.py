import numpy as np

# Загрузка данных
tensor = np.load('tensor.npy')
kernel = np.load('kernel.npy')
bias = np.load('bias.npy')
stride = int(open('task.csv').readlines()[1])

# Вычисление размеров выходного тензора
ch = (tensor.shape[1] - kernel.shape[0]) // stride + 1
cw = (tensor.shape[2] - kernel.shape[1]) // stride + 1

# Функция для применения свертки к одной части тензора
def apply_convolution(patch, kernel, bias):
    conv_result = np.sum(patch * kernel) + bias
    return conv_result

# Создание выходного тензора
my_tensor = np.zeros((tensor.shape[0], ch, cw, kernel.shape[3]))

# Применение свертки к каждой части входного тензора
for batch in range(tensor.shape[0]):
    for h_out in range(ch):
        for w_out in range(cw):
            h_start = h_out * stride
            h_end = h_start + kernel.shape[0]
            w_start = w_out * stride
            w_end = w_start + kernel.shape[1]
            patch = tensor[batch, h_start:h_end, w_start:w_end]
            for cout in range(kernel.shape[3]):
                my_tensor[batch, h_out, w_out, cout] = apply_convolution(patch, kernel[:, :, :, cout], bias[cout])

# Сохранение результата
np.save('seminar03_conv.npy', my_tensor, allow_pickle=False)