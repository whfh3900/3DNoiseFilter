# 3D Noise Filter
#### 3D 스캐너 또는 이미지 기반 재구성 기술로 얻은 포인트 클라우드는 종종 상당한 양의 노이즈와 아웃라이어로 손상됩니다. 여기서 아웃라이어란 포인트 클라우드 원본 표면(Surface)에서 멀리 떨어진 노이즈를 의미하며,  우리는 최근 포인트 클라우드에서 로컬 3D 모양 속성을 추정하기 위해 제안된 PCPNet에서 채택된 딥 러닝 아키텍처를 기반으로 접근합니다. 우리의 방법은 먼저 아웃라이어 샘플을 분류하고 버린 다음 원래의 깨끗한 표면에 노이즈 포인트를 투사하는 보정 벡터를 추정합니다. 즉, 첫 번째로 원본 표면에 멀리 떨어진 아웃라이어를 제거하는 두 번째로 원본 표면에 거친 부분을 평평하게 만들기 위해 포인트 클라우드를 옮기는 작업을 실행합니다.

본 소스코드는 포트폴리오 용으로 작성한 것이고, 실제 코드는https://github.com/mrakotosaon/pointcleannet 기반으로 작성하였다. 

![POINTCLEANNET 아키텍처](https://user-images.githubusercontent.com/48546917/83731606-c984b480-a685-11ea-9c57-8bb7d486adce.png)

# Install
pytorch
laspy
scipy

# Datasets and Models

# Process
## 1. Training

## 2. Inferencing
