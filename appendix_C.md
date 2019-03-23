# 拉格朗日对偶性

[TOC]



> 在约束最优化问题中，可以利用**拉格朗日对偶性**（Lagrange duality）将原始问题转换为对偶问题，通过求解对偶问题得到原始问题的解

## 1. 原始问题

- 假设$f(x), c_i(x), h_j(x)$是定义在$\mathbb{R}^n$上的连续可微函数，考虑约束最优化问题：

$$
\begin{aligned}
&\min\limits_{P\in\mathcal{C}}\quad f(x)\\
& \begin{array}{r@{\quad}r@{}l@{\quad}l}
\text{s.t.} & c_i(x) & \leq 0, &i=1,2,\dots,k\\
&h_j(x) & = 0 & j=1,2,\dots,l \\
\end{array}
\end{aligned}
$$

​	称此约束最优化问题为**原始最优化问题**或**原始问题**；

- **广义拉格朗日函数**（generalized Lagrange function）：

$$
L(x,\alpha,\beta)=f(x)+\sum_{i=1}^{k}\alpha_ic_i(x)+\sum_{j=1}^{l}\beta_jh_j(x)
$$

​	其中$x=(x^{(1)},x^{(2)},\dots,x^{(n)})^T \in\mathbb{R}^n$，$\alpha_i, \beta_j$是拉格朗日乘子，$\alpha_i\geq 0$：

- 考虑$x$的函数（下标$P$表示原始问题）：

$$
\theta_P(x)=\max\limits_{\alpha,\beta;\alpha_i\geq 0}L(x,\alpha,\beta)
$$

​	可以证明：
$$
\theta_P(x)=\left\{
\begin{aligned}
f(x), &\quad x满足原始问题约束 \\
+\infty, &\quad 其他 
\end{aligned}
\right.
$$

- 所有再考虑其极小化问题：

$$
\min\limits_{x}\theta_P(x)=\min\limits_{x}\max\limits_{\alpha,\beta;\alpha_i\geq 0}L(x,\alpha,\beta)
$$

​	与原始最优化问题等价（有相同的解），问题$\min\limits_{x}\max\limits_{\alpha,\beta;\alpha_i\geq 0}L(x,\alpha,\beta)​$称为**广义拉格朗日函数的极小极大问题**；

- 定义原始问题的最优值：

$$
p^*=\min\limits_{x}\theta_P(x)
$$

## 2. 对偶问题

- 定义$\alpha,\beta$的函数（下标$D$表示对偶问题）：

$$
\theta_D(\alpha,\beta)=\min\limits_{x}L(x,\alpha,\beta)
$$

- 再考虑其极大化：

$$
\max\limits_{\alpha,\beta;\alpha_i\geq 0}\theta_D(\alpha,\beta)=\max\limits_{\alpha,\beta;\alpha_i\geq 0}\min\limits_{x}L(x,\alpha,\beta)
$$

​	问题$\max\limits_{\alpha,\beta;\alpha_i\geq 0}\min\limits_{x}L(x,\alpha,\beta)$称为**广义拉格朗日函数的极大极小问题**



- 广义拉格朗日函数的极大极小问题可以表示为约束最优化问题：

$$
\begin{aligned}
\max\limits_{\alpha,\beta;\alpha_i\geq 0}\theta_D(\alpha,\beta) \quad&= \max\limits_{\alpha,\beta;\alpha_i\geq 0}\min\limits_{x}L(x,\alpha,\beta)\\

\text{s.t.} \quad&  \alpha_i  \geq 0, i=1,2,\dots,k\\

\end{aligned}
$$

​	称为原始问题的**对偶问题**；

- 定义对偶问题的最优值：

$$
d^*=\max\limits_{\alpha,\beta;\alpha_i\geq 0}\theta_D(\alpha,\beta)
$$

## 3. 原始问题和对偶问题的关系

- **定理1**：可证明如果原始问题和对偶问题都有最优值，则

$$
d^*=\max\limits_{\alpha,\beta;\alpha_i\geq 0}\theta_D(\alpha,\beta) \quad\leq\quad \min\limits_{x}\theta_P(x)=p^*
$$

- **推论1**：设$x^*​$和$\alpha^*, \beta^*​$分别是原始问题和对偶问题的<u>*可行解*</u>，且$d^*=p^*​$，则$x^*​$和$\alpha^*, \beta^*​$分别是原始问题和对偶问题的最优解；（某些条件下原始问题和对偶问题的最优值相等，可以<u>*用解对偶问题替代解原始问题*</u>）；



- **定理2**：假设函数$f(x)​$和$c_i(x)​$是凸函数，$h_j(x)​$是仿射函数；且不等式约束$c_i(x)​$是严格可行（存在$x​$，有$c_i(x)<0, \forall i​$），则存在$x^*,\alpha^*,\beta^*​$，使得$x^*​$是原始问题的解，$\alpha^*,\beta^*​$是对偶问题的解，且：

$$
d^*=p^*=L(x^*,\alpha^*,\beta^*)
$$



- **定理3**：假设函数$f(x)$和$c_i(x)$是凸函数，$h_j(x)$是仿射函数；且不等式约束$c_i(x)$是严格可行，则使得$x^*$是原始问题的解，$\alpha^*,\beta^*$是对偶问题的解的<u>*充分必要条件*</u>是满足下面的Karush-Kuhn-Tucker（KKT）条件：

$$
\begin{aligned}
\nabla_x L(x^*,\alpha^*,\beta^*) &= 0 \\
\nabla_\alpha L(x^*,\alpha^*,\beta^*) &= 0 \\
\nabla_\beta L(x^*,\alpha^*,\beta^*) &= 0 \\
\alpha_i^*c_i(x^*) &= 0, i=1,2,\dots,k \\
c_i(x^*) &\leq 0, i=1,2,\dots,k \\
\alpha_i^* &\geq 0, i=1,2,\dots,k \\
h_j(x^*) &=0, j=1,2,\dots,l
\end{aligned}
$$

​	其中$\alpha_i^*c_i(x^*) = 0, i=1,2,\dots,k$成为KKT的<u>*对偶互补条件*</u>，由此条件可知：如果$\alpha_i^*>0$，则$c_i(x^*)=0$