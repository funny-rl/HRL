
기존의 TempoRL은 FiGAR에 비해 설명 가능한 repeatation network를 가지지만 repeatation이 진행되는 동안 돌발 변수에 대처하기 어렵다는 단점이 존재한다.

action repeatation에 대해 매 step마다 실행되는 binary Termination function 아이디어


1. 매 step마다 $Q_{a}$와 $\mathbb{E_{a \in (A-a)}}[Q_{a}]$를 고려해 action을 계속할 지 중간에 멈출지 결정, 이렇게 할 경우 굳이 action 반복횟수 k를 정할 필요가 없을 수 있음.

    A. $Q_{a}$가 기댓값보다만 높으면 진행하는 것을 목표로 삼을 것이기 때문에 최적 정책에 맞지 않을 가능성이 높음. 


따라서 Termination function 학습을 위한 별도 MDP 모델링의 필요성 있음. 매 step $\beta(s_t, a_t):$ termination function을 계산한다. 출력 활성 함수는 sigmoid function이다. 


$$
\begin{align}

V(s_t) &= \sum_{a \in A} \pi(a_t|s_t)Q(s_t, a_t)\\

A(s_t, a_t) &= Q(s_t, a_t) - V(s_t) \\ \\

V_{term}(s_t) &= \beta(s_t, a_t)Q(s_t, a_t) + \left(1-\beta(s_t, a_t)\right)\mathbb{E}_{a_{t_{gu}} \in (A - a_t) }\left[\pi(a_{t_{gu}}|s_t)Q(s_t, a_{t_{gu}})\right] \\

A_{term}(s_t, a_t) &= Q(s_t, a_t) - V_{term}(s_t) \\

&= \left(1 - \beta(s_t, a_t)\right)\left(Q(s_t, a_t) - \mathbb{E}_{a_{t_{gu}} \in (A - a_t)}\left[\pi(a_{t_{gu}}|s_t)Q(s_t, a_{t_{gu}})\right] \right)  \\

\text{A2C}_{term} \text{loss} &= -\log{\beta(s_t, a_t)} \cdot A_{term}(s_t, a_t) \\


\end{align}
$$