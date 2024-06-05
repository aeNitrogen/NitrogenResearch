import torch


def single_input_builder(actions, observations):
    batch = actions.size(0)
    length = actions.size(1)
    features = actions.size(2) + observations.size(2)
    input = torch.zeros((batch, length, features))
    input[:, :, 0:] = actions
    input[:, :observations.size(1), actions.size(2):] = observations
    return input


def single_input(model, input):
    output = model(input)
    return output

def single_output(output):
    return output