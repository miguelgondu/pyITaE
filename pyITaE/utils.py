def to_json_writable(dict_):
    """
    TODO: embetter this function
    """
    new_dict = {str(k): v for k, v in dict_.items()}
    return new_dict
