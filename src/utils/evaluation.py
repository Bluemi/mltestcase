import torch.utils.data


def model_accuracy(model: torch.nn.Module, loader: torch.utils.data.DataLoader) -> float:
    sum_corrects = 0
    sum_all = 0

    with torch.no_grad():
        for data, labels in loader:
            prediction = model.forward_classify(data)

            label_prediction = prediction.argmax(dim=1)

            correct = torch.sum(torch.eq(label_prediction, labels))

            sum_corrects += correct.item()
            sum_all += labels.shape[0]

    return sum_corrects / sum_all
