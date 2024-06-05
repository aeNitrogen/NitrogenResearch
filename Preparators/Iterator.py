# iterate given model, data, target data, optimizer, methods, gpu batch size and loss criterion
import math


def iterate(model, data, target, optimizer, criterion, input_method, output_method, batch_size):
    # Iteration for training, weights only returned in iteration for evaluation.
    # Output method should not return weights.
    # Output cut defined by output method.
    for i in range(math.ceil(data.size(0) / batch_size)):
        start = i * batch_size
        end = min((i + 1) * batch_size, data.size(0))

        def closure():
            optimizer.zero_grad()
            output = output_method(model(input_method(data[start:end, :, :]), ))
            loss = criterion(output, target[start:end, :, :])
            loss.backward()
            return loss

        optimizer.step(closure)
