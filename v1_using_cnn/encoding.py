def oneHotEncoding(keys):
    # RIGHT+JUMP
    if 'Z' in keys and 'L' in keys:
        return [0, 0, 1]
    # JUMP
    elif 'Z' in keys:
        return [0, 1, 0]
    # RIGHT
    elif 'L' in keys:
        return [1, 0, 0]