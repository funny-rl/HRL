global optimal soluiton에서 벗어나더라도 의사결정횟수를 최대한 줄이는 것에 목적을 둔 open-loop action repetition

1. 만약 매 step 양의 보상 / 음의 보상이 주어지는 경우, repetition actor network가 과하게 lambda를 줄이는 모습을 보인다.

2. Repetition 횟수를 결정하는 네트워크의 학습 함수는 reptor-reptic 형태로, reptic 학습을 위해 별도의 reptic_Q를 추가적으로 도입했다. target은 critic의 target network와 보상 누적을 모두 활용한다. 다만 이 상황에서 reptic_Q가 repetition 횟수가 늘어날수록 저평가당한다고 생각했다. 왜냐하면 이론적으로 n-step-rep-Q는 1-step-rep-Q보다 같거나 작을 수 밖에 없기 때문이다. 다만 문제가 다른 곳에서 생겼다. 바로 기존 critic의 Q함수가 episode의 남은 step을 고려하지 못하는 문제가 있었기 때문이다. 따라서 n-step-rep-Q와 1-step-rep-Q의 값의 차이가 생각보다 작아지고 이로 인해 reptic의 target value가 n이 늘어날수록 엄청나게 작아지는 문제가 생겼기 때문이다. (기존 TempoRL / UTE는 critic의 Q를 학습할 때 time step 정보를 주지 않은 것 같다.) 따라서 이 문제를 해결하는 것을 우선했으며 time step 정보를 obs에 추가하니 확실히 step_rate (step / max_step)가 커질수록 전체 Q가 0에 어느정도 가까워지는 현상을 학습할 수 있었다. 

계획: critic의 Q를 재학습한 뒤에도 reptic_Q의 값이 n이 증가함에 따라 선호도가 낮아지는 문제가 발생하는 지 확인해야 한다. 만약 발생하지 않는다면 (생각보다 인공신경망이 1~n까지의 reptic_Q value를 오차의 점진적 증가 없이 잘 학습한다면) 이제 이론을 넘어 n이 증가했을 때의 별도 보상함수를 설계하는 것을 고려해볼 수 있다. 





알아낸 점

1. target critic network가 episode가 end에 가까워질수록 Q값을 정확히 예측하지 못한다. 
   
   1. 보상이 일정하게 매 step마다 주어지는 episodic 환경에서는 입력 state에 time을 주지 않으면 episode가 end에 가까워질수록 Q값이 0에 가까워지는 현상을 이해하지 못한다. (보상이 일정하지 않을 경우에 대해 고려해보아야 한다.)

2.  1번으로 인해 target critic network로 학습한 skip-Q network가 1-step repetition Q보다 n-step repetition Q에서 편향(과소평가)와 표준편차가 커지는 문제가 생긴다. 시간 정보를 추가함으로써 편향이 상당히 줄어드는 경향 (1-step과 n-step간의 차이가 해소됨)을 보이나 분산은 줄어들긴 해도 여전히 큰 영향을 미친다. (이론적인 형태를 충분히 어그러뜨릴 정도) 
    1.  분산을 줄이기 위해 네트워크 아키텍처 입력에 repetition step rate를 추가로 넣어봤는데 target q prediction의 variance가 줄지 않음. 

3. 왜 discount factor에 따라 n-step repetition의 대소관계가 변하는 거지..?


아이디어
1. 원래 목표는 1-step repetition Q에서 미분한 기울기를 활용한 1차함수를 통해 각 step에서의 q값을 어느정도 보정해줄 생각이었다.

2. repetition network의 출력이 n-step repetition Q가 아니라 n-step repetition V를 추정하게 하는 것은 어떨까? 
   1. 이렇게 하면 N-step repetition의 가치가 max(a')에 영향을 받는 문제가 해결됨. 그럼으로 인해 repetition에 대한 명확한 credit assignment가 가능해짐.
   2. Q를 추정하면 1-step보다 n-step이 같거나 작을수밖에 없었는데 V를 쓰면 이 대소관계를 해결할 수 있음(?)
   3. action marginalization에 따른 variance 감소.
   4. SMDP 이론과 일관되게 모델링 가능



[-0.25841124 -0.23844668 -0.22057004 -0.20480527 -0.191612   -0.17940852                                                                                                                                                                                                                
 -0.16743864 -0.15659971 -0.1451978 ]  

 [-0.28573917 -0.26449773 -0.24566315 -0.22906526 -0.21437758 -0.20111371
 -0.18821928 -0.17634661 -0.16473043]    