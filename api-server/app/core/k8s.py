from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException


def get_custom_objects_api() -> client.CustomObjectsApi:
    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()

    return client.CustomObjectsApi()