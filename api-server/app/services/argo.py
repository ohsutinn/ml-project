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
