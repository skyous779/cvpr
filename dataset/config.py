import logging

def get_config(name):

    config = {}  #字典

    if name.upper() == 'ARID':   #返回name中只包含大写的字符串
        config['num_classes'] = 6  #六??
    else:
        logging.error("Configs for dataset '{}'' not found".format(name))
        raise NotImplemented

    logging.debug("Target dataset: '{}', configs: {}".format(name.upper(), config))

    return config


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info(get_config("ARID"))
