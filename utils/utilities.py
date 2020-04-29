from time import time
import matplotlib.pyplot as plt
import numpy as np
import torch


def timer(func):
    """A decorator which records running time.

    Args:
        func (function): The function needs timekeeping.

    """
    def ticking(*args, **kwargs):
        start = time()
        value = func(*args, **kwargs)
        duration = time() - start
        print('ok@{}: finishes in '.format(func.__name__), end='')
        if duration // 3600 >= 1:
            print('{:.0f} hour(s) {:.0f} minute(s)'.format(duration // 3600, (duration % 3600) / 60))
        elif duration // 60 >= 1:
            print('{:.0f} minute(s) {:.0f} second(s)'.format(duration // 60, duration % 60))
        else:
            print('{:.3f} second(s)'.format(duration))
        return value
    return ticking


def type_error_msg(variable: str, invalid, valid_list):
    return 'Type {} should be a valid type in {}, but got type {}.'.format(variable, valid_list, type(invalid))


def value_error_msg(variable: str, invalid, valid_list, default=None):
    if default is None:
        error_string = '{variable} option should be a valid {variable} in {valid_list}, ' \
                       'but got {variable}={invalid}. '. \
            format(variable=variable, valid_list=valid_list, invalid=invalid)
    else:
        error_string = 'If specified, {variable} option should be a valid {variable} in {valid_list}, ' \
                       'but got {variable}={invalid}; ' \
                       'If no {variable} option is specified, ' \
                       'the default {variable} {default} is used.'. \
            format(variable=variable, valid_list=valid_list, invalid=invalid, default=default)
    return error_string


def format_path(template, name, delimiter):
    """Fills the placeholders to get the path.

    Args:
        template (str): A string that contains blanks to be filled.
        name (str): An option name of section defined in settings.txt.
        delimiter (str): The signal of a blank's beginning position.

    Returns:
        path (str): Path with name filled.

    """
    prefix, suffix = template.split(delimiter)
    return prefix % name + delimiter + suffix


def merge_last(last: dict, new: dict):
    """Merges an existing dictionary and a new dictionary.

    Args:
        last (dict): An existing dictionary.
        new (dict): The new dictionary to be added into the existing dictionary.

    Returns:
        result (dict): A merged(key-by-key) dictionary.

    """
    result = {}
    if not set(new.keys()).issubset(set(last.keys())):
        raise IndexError
    for elem, value in new.items():
        value = last[elem]
        if isinstance(value, np.ndarray):
            value = value.reshape(-1).tolist()
        result[elem] = value + new[elem]
    return result


def load_model(model, path):
    """Loads model.state_dict() from path. Implicitly returns model.

    Args:
        model (nn.Module): The model to load weights.
        path (str): The path to the saved model's state_dict.

    """
    model.load_state_dict(torch.load(path))


def save_model(model, path):
    """Saves model.state_dict() to path.

    Args:
        model (nn.Module): The model to save.
        path (str): The path to the saved model's state_dict.

    """
    torch.save(model.state_dict(), path)


def images_show(tags, images, title, index):
    """Generates an image consists of a triplet of imgs.

    Args:
        tags (list): The tags for every img.
        images (list): The tensor form of imgs.
        title (str): The title of the whole img.
        index (int): The indices of every img.

    Returns:
        fig (plt.figure)

    """
    fig = plt.figure(facecolor='white')
    fig.tight_layout()
    for i in range(len(tags)):
        ax = fig.add_subplot(1, 3, i + 1)
        if i == 0:
            ax.set_title('Anchor #{}'.format(tags[i]))
        elif i == 1:
            ax.set_title('Positive #{}'.format(tags[i]))
        else:
            ax.set_title('Negative #{}'.format(tags[i]))
        ax.axis('off')
        ax.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))  # from channel*height*width to height*width*channel
    fig.suptitle(title + str(index), fontsize=18, fontweight='bold')
    return fig

