# militech-2021-03
Reseach about adversarial patches suitable for military domains

## Research Info
 --> 여기다 우리 보고서 초록 넣기 / 아니면 이 문단 아예 삭제하기
 
## Repository Info
본 Repository는 EAVAISE의 Adversarial YOLO 구현을 기반으로 한다 : https://gitlab.com/EAVISE/adversarial-yolo   
본 Repository는 이 연구의 보고서를 따른다 : 보고서 링크
 
## How to Use
### requirement
Python 3.6.버전을 사용한다. 또한 Pytorch가 설치되어 있어야한다 : https://pytorch.org/  
학습과정을 시각화하기 위해서, 아래와 같이 pip을 이용하여 tensorboardX를 설치 할 수 있다. 
```python
pip install tensorboardX tensorboard
```  
학습에 필요한 YOLOv2 MS COCO weights 및 INRIA dataset은 Repository에 모두 저장되어 있다.

### Training Patch
train_patch.py를 수정하여 적대적패치의 시작 이미지, 유사성을 대조할 이미지, Loss 등을 변경할 수 있다.  
아래의 커맨드를 이용하여 패치를 학습시킨다
```python
python train_patch.py 
```  
 
 
