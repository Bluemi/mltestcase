import torch.utils.data

from utils import fourier_transform_2d, cosine_transform_2d


def model_accuracy(model: torch.nn.Module, loader: torch.utils.data.DataLoader, use_ft=False) -> float:
    sum_corrects = 0
    sum_all = 0

    with torch.no_grad():
        for data, labels in loader:
            if use_ft == 'fft':
                data = fourier_transform_2d(data)
            elif use_ft == 'dct':
                data = cosine_transform_2d(data)
            prediction = model.forward_classify(data)

            label_prediction = prediction.argmax(dim=1)

            correct = torch.sum(torch.eq(label_prediction, labels))

            sum_corrects += correct.item()
            sum_all += labels.shape[0]

    return sum_corrects / sum_all
