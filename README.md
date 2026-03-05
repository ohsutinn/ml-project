# Kubernetes 기반 데이터셋 업로드–학습–서빙 통합 ML 플랫폼

# 프로젝트 목표
사용자가 대용량(CSV/Excel 최대 2GB) 데이터를 업로드하면 이를 안전하게 저장하고, 선택한 데이터셋으로 모델 학습을 수행한 뒤 성능 지표와 함께 결과를 제공하는 플랫폼을 구축한다.  
학습된 모델은 버전 관리(삭제/복원 포함)하며, 모델별로 다른 입·출력 구조를 API로 추론(서빙)할 수 있도록 설계하고, 신규 데이터 반영을 통한 재학습까지 이어질 수 있는 구조를 목표로 한다.

---

# 시스템 아키텍처
<img width="1832" height="1392" alt="ml-project 아키텍처" src="https://github.com/user-attachments/assets/d6e2cafc-43c5-4209-aa6f-78db370aa327" />

---

# ERD
<img width="1780" height="1142" alt="ML" src="https://github.com/user-attachments/assets/9f58bf58-263e-4ced-8214-0fb13b48eab0" />

---

# 적용 기술

### 프레임워크 / 라이브러리
![Python](https://img.shields.io/badge/Python-3.11.14-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.120.4-009688?style=flat-square&logo=fastapi&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0.44-D71F00?style=flat-square&logo=sqlalchemy&logoColor=white)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![TFDV](https://img.shields.io/badge/TensorFlow%20Data%20Validation-1.17.0-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)

![MLflow](https://img.shields.io/badge/MLflow-3.8.0-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![Weights%20%26%20Biases](https://img.shields.io/badge/W%26B-0.23.1-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)

![BentoML](https://img.shields.io/badge/BentoML-1.4.31-000000?style=flat-square&logo=bentoml&logoColor=white)

### 데이터베이스 / 스토리지
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18.3-336791?style=flat-square&logo=postgresql&logoColor=white)
![MinIO](https://img.shields.io/badge/MinIO-7.2.18-C72E49?style=flat-square&logo=minio&logoColor=white)

### 외부 연동
![Weights%20%26%20Biases](https://img.shields.io/badge/W%26B-Experiment%20Tracking-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)

### 인프라
![Kubernetes](https://img.shields.io/badge/Kubernetes-Cluster-326CE5?style=flat-square&logo=kubernetes&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=flat-square&logo=docker&logoColor=white)
![NGINX%20Ingress](https://img.shields.io/badge/NGINX%20Ingress-Controller-009639?style=flat-square&logo=nginx&logoColor=white)
![Argo%20Workflows](https://img.shields.io/badge/Argo%20Workflows-3.7.6-EF7B4D?style=flat-square&logo=argo&logoColor=white)

---

# 주요 기능
### 1) 데이터 업로드 & 데이터셋 관리
- FastAPI로 파일 업로드(최대 2GB) 후 MinIO 저장
- Dataset / DatasetVersion 생성 및 버전 목록 조회
- DatasetVersion 상태 관리(PENDING/PROFILING/READY/FAILED/DELETED 등)

### 2) 데이터 검증/프로파일링
- TFDV로 통계 생성 및 스키마 추론
- 스키마 기반 품질 검증 및 anomaly 요약
- 베이스라인 통계/스키마 저장, 이후 버전 검증 시 baseline 비교(드리프트/분포 변화 포함)

### 3) 학습 파이프라인 오케스트레이션
- Argo Workflow 트리거 코드 제공
- 전처리/HPO/등록용 클라이언트 스텝 구현
- 스텝 간 아티팩트/파라미터 전달을 위한 output 파일 생성

### 4) 하이퍼파라미터 튜닝(HPO)
- W&B Sweeps 로 탐색 공간 정의 및 실험 수행
- 최고 성능 run 선택 및 best hparams/model 아티팩트 저장
- 메트릭 비교/요약(val_rmse 또는 val_accuracy 기준)

### 5) 모델 레지스트리 & 버전 관리
- MLflow Tracking에 params/metrics/artifacts 로그
- MLflow Registry에 모델 등록 및 버전 생성
- 모델명/버전/run_id를 파이프라인 출력으로 제공

### 6) 수동 배포 승인
- 별도 promote API로 배포 워크플로우(Argo) 트리거
- 승인 전에는 레지스트리 상태로 유지

### 7) 모델 서빙 & 온라인 추론
- BentoML 서비스로 REST 기반 예측 API 제공
- 요청 기반 실시간 예측
- MLflow 모델 로드 및 optional preprocessor 적용

### 8) 보안/권한/멀티테넌시
- 네임스페이스 분리, RBAC(Role/RoleBinding) 기반 권한 제어
- 시크릿/환경변수 관리(K8s Secret/ConfigMap) 및 안전한 설정 주입


## 기능별 시퀀스 다이어그램
### 데이터셋 업로드
<img width="6790" height="2205" alt="image" src="https://github.com/user-attachments/assets/7b9e5be0-0e98-4783-a28e-35ca5435dc5f" />

### 데이터셋 베이스라인 생성
<img width="8192" height="2924" alt="image" src="https://github.com/user-attachments/assets/ff6950fb-c862-4cd1-b45d-de52815f5d72" />

### 모델 학습
<img width="8192" height="3806" alt="image" src="https://github.com/user-attachments/assets/082b3748-e294-4ebd-a653-ef35bfaab638" />

### 모델 서빙
<img width="8192" height="2465" alt="image" src="https://github.com/user-attachments/assets/654b62a7-a0d6-43f0-a169-80475bc64082" />

---

# 기술적 의사결정
<ul>
  <li><a href="https://github.com/ohsutinn/ml-project/wiki/%EC%98%A4%EB%B8%8C%EC%A0%9D%ED%8A%B8-%EC%8A%A4%ED%86%A0%EB%A6%AC%EC%A7%80-%EC%84%A0%ED%83%9D">오브젝트 스토리지 선택</a></li>
</ul>
<ul>
  <li><a href="https://github.com/ohsutinn/ml-project/wiki/%EC%9B%8C%ED%81%AC%ED%94%8C%EB%A1%9C%EC%9A%B0-%EC%98%A4%EC%BC%80%EC%8A%A4%ED%8A%B8%EB%A0%88%EC%9D%B4%EC%85%98-%EC%84%A0%ED%83%9D">워크플로우 오케스트레이션 선택</a></li>
</ul>
<ul>
  <li><a href="https://github.com/ohsutinn/ml-project/wiki/Hyperparameter-Optimization-%EB%8F%84%EC%9E%85-%EB%B0%8F-%EC%84%A0%ED%83%9D">Hyperparameter Optimization 도입 및 선택</a></li>
</ul>

---

# 트러블 슈팅
<ul>
  <li><a href="https://github.com/ohsutinn/ml-project/wiki/Mac-M2-%ED%99%98%EA%B2%BD%EC%97%90%EC%84%9C-TFDV-%EC%84%A4%EC%B9%98-%EC%8B%A4%ED%96%89-%EC%8B%A4%ED%8C%A8-%E2%86%92-x86_64-%EC%A0%84%EC%9A%A9-%EB%85%B8%EB%93%9C%EB%A1%9C-%EB%B6%84%EB%A6%AC">Mac M2 환경에서 TFDV 설치 실행 실패 → x86_64 전용 노드로 분리</a></li>
</ul>
<ul>
  <li><a href="https://github.com/ohsutinn/ml-project/wiki/CoreDNS-%EC%97%85%EC%8A%A4%ED%8A%B8%EB%A6%BC-DNS-%EC%98%A4%EB%A5%98%EB%A1%9C-%EC%9D%B8%ED%95%9C-Pod-DNS-%EC%9E%A5%EC%95%A0-%ED%95%B4%EA%B2%B0">CoreDNS 업스트림 DNS 오류로 인한 Pod DNS 장애 해결</a></li>
</ul>
<ul>
  <li><a href="https://github.com/ohsutinn/ml-project/wiki/DiskPressure-%ED%8A%B8%EB%9F%AC%EB%B8%94%EC%8A%88%ED%8C%85">DiskPressure 문제 해결</a></li>
</ul>
