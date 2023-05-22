# converted args to a class
class Args(object):
    # convert dictionary to class
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)