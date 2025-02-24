from models.supervised import FCN, ConvTran


def Model_factory(config):
    if config['Model_Type'][0] == 'FCN':
        model = FCN.FCN(config, num_classes=config['num_labels'])
    if config['Model_Type'][0] == 'ConvTran':
        model = ConvTran.ConvTran(config, num_classes=config['num_labels'])
    if config['Model_Type'][0] == 'ConvTran_max':
        model = ConvTran.ConvTran_max(config, num_classes=config['num_labels'])
    if config['Model_Type'][0] == 'TimeCNN':
        model = ConvTran.Time_CNN(config, num_classes=config['num_labels'])
    return model
