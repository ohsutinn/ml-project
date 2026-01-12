import json
import uuid

from app.core.k8s import get_custom_objects_api


ARGO_NAMESPACE = "argo"
ARGO_DV_TEMPLATE_NAME = "dv-initial-baseline"
ARGO_TRAIN_TEMPLATE_NAME = "model-train"


def create_dv_initial_baseline_workflow(payload: dict) -> str:
    custom_api = get_custom_objects_api()

    run_id = uuid.uuid4().hex[:8]
    workflow_name = f"dv-initial-baseline-{run_id}"

    workflow_manifest = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {
            "name": workflow_name,
            "namespace": ARGO_NAMESPACE,
            "labels": {
                "app": "automl-api",
                "dv-mode": str(payload.get("mode", "")),
                "dataset-id": str(payload.get("dataset_id", "")),
                "dataset-version-id": str(payload.get("dataset_version_id", "")),
            },
        },
        "spec": {
            "workflowTemplateRef": {
                "name": ARGO_DV_TEMPLATE_NAME,
            },
            "arguments": {
                "parameters": [
                    {
                        "name": "payload-json",
                        "value": json.dumps(payload, ensure_ascii=False),
                    }
                ]
            },
        },
    }

    # CRD 생성
    custom_api.create_namespaced_custom_object(
        group="argoproj.io",
        version="v1alpha1",
        namespace=ARGO_NAMESPACE,
        plural="workflows",
        body=workflow_manifest,
    )

    return workflow_name


def create_model_train_workflow(payload: dict) -> str:
    custom_api = get_custom_objects_api()

    run_id = uuid.uuid4().hex[:8]
    workflow_name = f"model-train-{run_id}"

    workflow_manifest = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {
            "name": workflow_name,
            "namespace": ARGO_NAMESPACE,
            "labels": {
                "app": "automl-api",
                "type": "model-train",
                "model-id": str(payload.get("model_id", "")),
                "dataset-id": str(payload.get("dataset_id", "")),
                "dataset-version-id": str(payload.get("dataset_version_id", "")),
            },
        },
        "spec": {
            "workflowTemplateRef": {
                "name": ARGO_TRAIN_TEMPLATE_NAME,
            },
            "arguments": {
                "parameters": [
                    {
                        "name": "payload-json",
                        "value": json.dumps(payload, ensure_ascii=False),
                    }
                ]
            },
        },
    }

    custom_api.create_namespaced_custom_object(
        group="argoproj.io",
        version="v1alpha1",
        namespace=ARGO_NAMESPACE,
        plural="workflows",
        body=workflow_manifest,
    )

    return workflow_name


ARGO_DEPLOY_TEMPLATE_NAME = "model-deploy" 

def create_model_deploy_workflow(payload: dict) -> str:
    """
    배포 워크플로우(WorkflowTemplate=model-deploy) 트리거
    payload에는 model_id, registered_model_name, alias, tracking uri 등 포함
    """
    custom_api = get_custom_objects_api()

    run_id = uuid.uuid4().hex[:8]
    workflow_name = f"model-deploy-{run_id}"

    # 필수값 체크(원하면 더 엄격하게 해도 됨)
    required = [
        "model_id",
        "registered_model_name",
        "src_alias",
        "dst_alias",
        "serving_namespace",
        "deploy_name",
        "serving_image",
        "bento_signature_method",
    ]
    missing = [k for k in required if not payload.get(k)]
    if missing:
        raise ValueError(f"create_model_deploy_workflow payload missing: {missing}")

    workflow_manifest = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {
            "name": workflow_name,
            "namespace": ARGO_NAMESPACE,
            "labels": {
                "app": "automl-api",
                "type": "model-deploy",
                "model-id": str(payload.get("model_id", "")),
                "deploy-name": str(payload.get("deploy_name", "")),
                "dst-alias": str(payload.get("dst_alias", "")),
            },
        },
        "spec": {
            "workflowTemplateRef": {"name": ARGO_DEPLOY_TEMPLATE_NAME},
            "arguments": {
                "parameters": [
                    {"name": "model_id", "value": str(payload["model_id"])},
                    {"name": "registered_model_name", "value": str(payload["registered_model_name"])},
                    {"name": "src_alias", "value": str(payload["src_alias"])},
                    {"name": "dst_alias", "value": str(payload["dst_alias"])},
                    {"name": "serving_namespace", "value": str(payload["serving_namespace"])},
                    {"name": "deploy_name", "value": str(payload["deploy_name"])},
                    {"name": "serving_image", "value": str(payload["serving_image"])},
                    {"name": "bento_signature_method", "value": str(payload["bento_signature_method"])},
                ]
            },
        },
    }

    custom_api.create_namespaced_custom_object(
        group="argoproj.io",
        version="v1alpha1",
        namespace=ARGO_NAMESPACE,
        plural="workflows",
        body=workflow_manifest,
    )

    return workflow_name