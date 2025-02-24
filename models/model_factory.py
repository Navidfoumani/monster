from models.deep_learning import FCN, ConvTran


def Model_factory(config):
    if config['Model_Type'][0] == 'FCN':
        model = FCN.FCN(config, num_classes=config['num_labels'])
    if config['Model_Type'][0] == 'ConvTran':
        model = ConvTran.ConvTran(config, num_classes=config['num_labels'])
    return model
