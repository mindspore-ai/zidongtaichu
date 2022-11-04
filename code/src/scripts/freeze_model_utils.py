
import os
import stat
import yaml
from tk.src.log.log import logger

CONN_WITH = '.'
FREEZE_KEY = 'freeze'

def freeze_model(model, freeze_layers):
    """
    冻结网络指定部分
    :param model:网络模型
    :param freeze_layers:冻结部分, 值是一个字符串列表
    """
    if not freeze_layers:
        logger.info('freeze_layers is empty, no layers in model will be frozen.')
        return

    if not isinstance(freeze_layers, list):
        freeze_layers = list(freeze_layers)

    layer_list = []
    for layer in freeze_layers:
        layer_list.append({'layer': layer, 'exist': False})

    logger.info('freeze model start.')

    for name, param in get_name_and_param_of_model(model):
        for value in layer_list:
            if not isinstance(value.get('layer'), str):
                raise ValueError('freeze layer is not str, freeze layer: %s' % freeze_layers)
            if name.startswith(value.get('layer')):
                param.requires_grad = False
                value['exist'] = True
    for value in layer_list:
        if not value['exist']:
            logger.warning('layer: %s is not exist.', value.get('layer'))

    logger.info("freeze model finish.")


def get_freeze_layers(model_config_path):
    """
    从model config配置文件中, 解析出mindspore/pytorch能够识别的需要冻结的网络层
    :param model_config_path: model config配置文件本地绝对路径
    :return: 需要冻结的网络层名称集合
    """
    # 软链接校验
    if os.path.islink(model_config_path):
        logger.warning('detect link path, stop parsing free configs from model config file.')
        return []

    # 路径真实性校验
    if not os.path.exists(model_config_path):
        logger.error('model config path is not exist.')
        raise FileExistsError

    try:
        content = read_file(model_config_path)
    except IOError as ex:
        logger.error('exception occurred when reading model config file, detail error message: %s', ex)
        raise ex

    if FREEZE_KEY not in content.keys():
        logger.error('no [freeze] config found in model config file, no layers will be frozen.')
        return []

    freeze_info_dict = content.get(FREEZE_KEY)
    if freeze_info_dict is None:
        logger.error('[freeze] attribute is empty in model config file, check model config file.')
        return []

    expanded_dict = expand_dict(freeze_info_dict)
    res = split_vals_with_same_key(expanded_dict)
    return res


def read_file(model_config_path):
    """
    读取配置文件
    """
    flags = os.O_RDWR | os.O_CREAT  # 允许读写, 文件不存在时新建
    modes = stat.S_IWUSR | stat.S_IRUSR  # 所有者读写

    with os.fdopen(os.open(model_config_path, flags, modes), 'rb') as file:
        content = yaml.safe_load(file)

    return content


def get_name_and_param_of_model(model):
    """
    自动区分pytorch/mindspore框架, 获取模型的参数名称和参数值
    :param model:网络模型
    :return:参数名称及参数值对
    """
    if hasattr(model, "parameters_and_names"):
        for _, param in model.parameters_and_names():
            name = param.name
            yield name, param
    elif hasattr(model, "named_parameters"):
        for name, param in model.named_parameters():
            yield name, param
    else:
        raise RuntimeError("unsupported backend framework, only support mindspore and pytorch.")


def expand_dict(dict_info):
    """
    将网络冻结配置解析出的字典平铺化
    :param dict_info: 平铺前的配置字典
    :return: 平铺后的配置字典
    """
    common_prefix_dict = dict()

    for key, val in dict_info.items():
        if key is None:
            logger.error('find [none] key from [freeze] config in model config file, '
                         'config is ignored, check model config file.')
            continue

        if val is None:
            logger.error('attribute of key: [%s] is none, config is ignored, check model config file.', str(key))
            continue

        if isinstance(val, dict):
            val = expand_dict(val)
            common_prefix_dict.update(get_prefix_dict(dict_info=val, prefix_str=str(key)))
        else:
            common_prefix_dict.update({str(key): [_val_item for _val_item in str(val).split(' ')]})

    return common_prefix_dict


def split_vals_with_same_key(expanded_dict):
    """
    对同一前缀下的多个子名称进行拆分
    :param expanded_dict: 平铺后的字典
    :return: 拆分后的完整名称列表
    """
    res = []
    for key, val in expanded_dict.items():
        for _val_item in val:
            res.append(str(key) + CONN_WITH + str(_val_item))
    return res


def get_prefix_dict(dict_info, prefix_str):
    """
    获取包含前缀字典(多层嵌套使用)
    :param dict_info: 配置字典
    :param prefix_str: 前缀信息
    :return: 拼接前缀后的字典
    """
    return {prefix_str + CONN_WITH + str(k): v for k, v in dict_info.items()}