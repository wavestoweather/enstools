def fix_attributes(data_array, suffix):
    """
    Fix a Data Array attributes adding a suffix to several of them.
    :param data_array:
    :param suffix:
    :return:
    """
    # Rewrite all the attributes:
    data_array.name = f"{data_array.name}_{suffix}"

    new_attributes = {
        "long_name": f"{suffix} of %s",
        "short_name": f"%s_{suffix}",
        "standard_name": f" %s_{suffix}",
    }
    for key, substitution in new_attributes.items():
        if key in data_array.attrs:
            data_array.attrs[key] = substitution % data_array.attrs[key]

    attributes_to_delete = ["code", "table"]

    for attr in attributes_to_delete:
        if attr in data_array.attrs:
            del (data_array.attrs[attr])
    return data_array
