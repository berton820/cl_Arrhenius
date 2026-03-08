# A Physics-Informed Neural Network Initialized by Arrhenius Priors for Robust Extrapolation of Hot Deformation Flow Stress in TC4 Alloy

## 中文题目建议
基于 Arrhenius 物理先验初始化神经网络的 TC4 钛合金热变形流变应力稳健外推建模

## Highlights
- 提出一种面向热变形本构建模的 `NN-PhysicsInit` 框架，将 Arrhenius 本构关系生成的合成样本用于神经网络预训练，再利用真实实验数据进行微调。
- 在温度外推与应变速率外推两类更贴近工程应用的任务上，`NN-PhysicsInit` 同时保持了物理模型的外推稳定性与数据驱动模型的拟合能力。
- 相比纯数据驱动神经网络，所提方法显著抑制了未见工况下的预测震荡与相对误差恶化问题。
- 研究结果表明，物理先验并非只是提升训练精度，更重要的是改善跨域泛化与工程可用性。

## Abstract
Accurate constitutive modeling of hot deformation behavior is essential for process design, simulation, and optimization of titanium alloys. Traditional Arrhenius-type constitutive equations exhibit good physical interpretability and extrapolation stability, but their accuracy is often limited when the deformation behavior becomes strongly nonlinear. In contrast, purely data-driven neural networks usually achieve high fitting accuracy within the training domain, yet they may suffer from poor extrapolation and oscillatory predictions under unseen conditions. To address this contradiction, this study proposes a physics-informed neural network framework, termed `NN-PhysicsInit`, for robust flow stress prediction of TC4 alloy under hot deformation. The core idea is to use a strain-compensated Arrhenius model to generate synthetic samples over a broad thermo-mechanical domain, pretrain a neural network with these physically consistent data, and then finetune the network using real experimental observations.

Two representative extrapolation tasks were designed to evaluate the framework. In the temperature extrapolation task, data at `800-980 °C` were used for training and data at `1010 °C` were reserved for testing. In the strain-rate extrapolation task, data at `0.001-1 s^-1` were used for training and data at `10 s^-1` were used for testing. Four methods were compared, including the conventional Arrhenius model, a direct neural network (`NN-Direct`), an Arrhenius-pretrained-only neural network (`NN-Arrhenius-Only`), and the proposed `NN-PhysicsInit`. The results show that `NN-Direct` achieves the highest in-domain fitting accuracy but suffers from severe extrapolation degradation, whereas `NN-PhysicsInit` consistently obtains the best balance between fitting accuracy and extrapolation robustness. In the temperature extrapolation task, `NN-PhysicsInit` reduces the extrapolation AARE to `12.93%` and the RMSE to `10.27 MPa`, outperforming Arrhenius (`27.49%`, `15.45 MPa`) and `NN-Direct` (`36.34%`, `15.14 MPa`). In the strain-rate extrapolation task, `NN-PhysicsInit` further reduces the extrapolation AARE to `9.09%` and the RMSE to `24.28 MPa`, significantly better than Arrhenius (`17.84%`, `41.32 MPa`) and `NN-Direct` (`21.40%`, `49.19 MPa`).

These results demonstrate that embedding physics priors into neural network initialization is an effective route for building reliable constitutive models under limited data and unseen deformation conditions. The proposed framework provides a promising strategy for hot processing simulation, digital process design, and intelligent constitutive modeling of titanium alloys.

## Keywords
TC4 alloy; hot deformation; Arrhenius constitutive model; physics-informed neural network; extrapolation; flow stress prediction

## 1. Introduction
TC4 titanium alloy is widely used in aerospace, high-end equipment, and energy-related applications because of its high specific strength, excellent corrosion resistance, and good comprehensive mechanical properties. During hot working, however, its flow stress behavior is strongly coupled with temperature, strain rate, and strain. Reliable constitutive models are therefore indispensable for process window design, finite element simulation, and microstructure-process-property integration.

At present, Arrhenius-type constitutive equations represented by the Sellars-Tegart hyperbolic-sine framework remain among the most commonly used approaches for hot deformation modeling. Their main advantages are clear physical meaning, compact mathematical form, and relatively good extrapolation stability. Nevertheless, once the deformation response exhibits strong nonlinearity, stage-dependent softening, or local irregularity, the prediction accuracy of such models can become insufficient. On the other hand, neural networks have powerful nonlinear fitting capability and often produce extremely high in-domain accuracy. Their main weakness is that they tend to learn statistical interpolation rather than physically constrained evolution laws, which makes them vulnerable to extrapolation failure in untrained regimes.

This contradiction is highly relevant in practical hot working. In industrial scenarios, researchers are often interested not only in the measured deformation conditions, but also in neighboring or even unmeasured temperature and strain-rate windows. A constitutive model that performs well only inside the training domain is therefore of limited engineering value. The central motivation of this work is to build a hybrid modeling framework that preserves the extrapolation stability of a physics-based constitutive law while leveraging the nonlinear representation power of deep neural networks.

Based on this motivation, this study develops a physics-prior-initialized neural network framework using Arrhenius-generated synthetic data and validates it on two challenging extrapolation tasks derived from TC4 hot deformation data. Compared with standard model comparison studies that mainly focus on interpolation accuracy, the present work emphasizes cross-domain robustness, metric reliability, and engineering-oriented model behavior.

## 2. Research Motivation
The motivation of this study can be summarized in four aspects.

First, traditional Arrhenius constitutive models have strong physical grounding, but their expressive capacity is limited when dealing with complex nonlinear flow stress evolution across wide thermo-mechanical ranges.

Second, purely data-driven neural networks can produce excellent fitting performance on training data, yet such performance may be misleading because high interpolation accuracy does not guarantee stable extrapolation under unseen temperatures or strain rates.

Third, hot deformation experiments are costly and time-consuming. In many realistic settings, only sparse measurements are available, while extrapolation to untested conditions is still required for process design and simulation.

Fourth, current materials informatics studies often emphasize predictive accuracy but pay insufficient attention to whether the learned model remains physically plausible outside the experimental domain. This gap motivates the introduction of a physically meaningful pretraining stage.

## 3. Main Innovations and Expected Benefits
### 3.1 Main innovations
1. A hybrid framework combining Arrhenius constitutive priors with neural network learning is proposed. Instead of directly training on sparse real data, the method first learns the broad physical trend from synthetic data and then adapts to real measurements.
2. Two complementary extrapolation scenarios are designed, namely temperature extrapolation and strain-rate extrapolation. This allows the method to be evaluated not only for fitting capability but also for cross-domain generalization.
3. For the temperature extrapolation task, a more engineering-oriented evaluation strategy is introduced by using a robust AARE definition with an effective strain region and denominator floor, reducing the artificial inflation of relative error in low-stress segments.
4. The data preprocessing strategy is improved by introducing shape-preserving interpolation (`PCHIP`) to mitigate overshoot and oscillation in low-stress regions.
5. Comparative evidence is provided through four models, showing that the proposed framework occupies a middle ground between pure physics and pure data fitting, thereby offering both accuracy and robustness.

### 3.2 Expected benefits
1. The proposed framework can reduce dependence on exhaustive hot compression experiments and improve model utility under partially observed deformation domains.
2. The resulting constitutive model is more suitable for process optimization and finite element simulation because it is less likely to produce unstable or oscillatory predictions in extrapolated regimes.
3. The method provides a transferable paradigm for other alloys and thermo-mechanical problems where data scarcity and extrapolation reliability are both critical.
4. From an engineering perspective, improvements in AARE and RMSE under unseen conditions directly enhance trustworthiness in digital process design.

## 4. Materials and Methods
### 4.1 Experimental dataset
The dataset used in this study was collected from hot deformation experiments of TC4 alloy and stored in `TC4_0219.xlsx`. The full dataset contains `39109` raw points covering temperatures from `800 °C` to `1010 °C` and strain rates from `0.001 s^-1` to `10 s^-1`. The raw stress-strain curves were further interpolated to a discrete strain grid to support constitutive parameter identification and model comparison under unified sampling conditions.

### 4.2 Two extrapolation tasks
To verify the robustness of the proposed method, two extrapolation experiments were designed.

In the temperature extrapolation task, samples at `800, 850, 900, 950, 980 °C` were used as the training set, while all data at `1010 °C` were treated as the test set. After interpolation, the training set contained `325` discrete points and the test set contained `65` points.

In the strain-rate extrapolation task, samples at `0.001, 0.01, 0.1, 1.0 s^-1` were used for training, while data at `10 s^-1` were reserved for testing. After interpolation, the training set contained `310` discrete points and the test set contained `78` points.

These two tasks represent two distinct types of generalization difficulty. Temperature extrapolation tests sensitivity to thermal activation and deformation mechanism transfer, whereas strain-rate extrapolation tests the ability to generalize across an order-of-magnitude change in loading rate.

### 4.3 Compared models
Four models were compared in this study.

`Arrhenius`: a conventional strain-compensated Arrhenius constitutive model used as the physics-based baseline.

`NN-Direct`: a neural network trained only on real experimental data, representing a purely data-driven route.

`NN-Arrhenius-Only`: a neural network trained only on Arrhenius-generated synthetic data, used to test whether the network can recover the physical structure itself.

`NN-PhysicsInit`: the proposed model, which is first pretrained on Arrhenius-generated synthetic data and then finetuned on real experimental data.

### 4.4 Physics-prior-initialized neural network
The proposed framework consists of three stages.

First, a global Arrhenius model is calibrated from the training-domain data. The identified constitutive parameters are strain-dependent and smoothed through polynomial compensation, enabling stress prediction over a broader thermo-mechanical range.

Second, the calibrated Arrhenius model is used to generate a large synthetic dataset over an expanded domain of temperature, strain rate, and strain. These synthetic samples provide the neural network with physically meaningful initialization and teach the network the global monotonic trends implied by the constitutive law.

Third, the pretrained neural network is finetuned using real experimental samples. In the optimized temperature extrapolation notebook, the network uses features `[1000/T, ln(strain rate), strain]` and predicts `log(stress)` rather than raw stress, which improves relative-error behavior and stabilizes learning.

### 4.5 Evaluation metrics
The main metrics include the correlation coefficient `R`, coefficient of determination `R^2`, average absolute relative error `AARE`, and root mean square error `RMSE`.

For the temperature extrapolation experiment, a robust AARE was additionally introduced to reduce metric distortion in low-stress regions. Specifically, an effective strain interval was used and the denominator in the relative-error term was floored by a minimum stress threshold. Raw AARE was still retained for comparison.

## 5. Results
### 5.1 Temperature extrapolation at 1010 °C
The temperature extrapolation results clearly reveal the difference between fitting ability and extrapolation robustness. The direct neural network achieves the best in-domain fitting performance (`R^2 = 0.9958`, `AARE = 3.28%`), but its extrapolation error at `1010 °C` rises sharply (`AARE = 36.34%`). This indicates that high training accuracy alone is not sufficient for reliable constitutive prediction beyond the observed temperature range.

By contrast, the proposed `NN-PhysicsInit` obtains a more balanced performance. Although its extrapolation `R^2` (`0.8892`) is not the highest among all models, it achieves the best extrapolation AARE (`12.93%`) and RMSE (`10.27 MPa`). This suggests that the model captures the overall physical tendency more stably and avoids the relative-error explosion seen in purely data-driven prediction.

The full comparison is summarized below.

| Method | Dataset | R | R² | AARE (%) | RMSE (MPa) | AARE_raw (%) | N |
|---|---|---:|---:|---:|---:|---:|---:|
| Arrhenius | Training (800-980 °C) | 0.9784 | 0.9572 | 16.51 | 22.50 | 16.60 | 281 |
| Arrhenius | Extrapolation (1010 °C) | 0.9407 | 0.8849 | 27.49 | 15.45 | 28.38 | 48 |
| NN-Direct | Training (800-980 °C) | 0.9979 | 0.9958 | 3.28 | 6.28 | 3.29 | 281 |
| NN-Direct | Extrapolation (1010 °C) | 0.9741 | 0.9489 | 36.34 | 15.14 | 38.12 | 48 |
| NN-Arrhenius-Only | Training (800-980 °C) | 0.9783 | 0.9571 | 16.58 | 22.55 | 16.67 | 281 |
| NN-Arrhenius-Only | Extrapolation (1010 °C) | 0.9410 | 0.8855 | 27.49 | 15.43 | 28.38 | 48 |
| NN-PhysicsInit | Training (800-980 °C) | 0.9933 | 0.9866 | 7.14 | 11.32 | 7.15 | 281 |
| NN-PhysicsInit | Extrapolation (1010 °C) | 0.9430 | 0.8892 | 12.93 | 10.27 | 13.28 | 48 |

An additional sensitivity analysis showed that the extrapolation AARE of `NN-PhysicsInit` at `1010 °C` remained in a narrow band when the minimum effective strain threshold was varied, indicating that the model improvement is not a metric artifact but a stable trend.

### 5.2 Strain-rate extrapolation from 1 s^-1 to 10 s^-1
The strain-rate extrapolation task is more challenging because it requires transfer across an order-of-magnitude increase in deformation rate. Under this task, `NN-Direct` again performs best inside the training domain (`R^2 = 0.9998`, `AARE = 1.67%`) but degrades severely in extrapolation (`R^2 = 0.8985`, `AARE = 21.40%`, `RMSE = 49.19 MPa`).

The proposed `NN-PhysicsInit` shows the best overall generalization. Its extrapolation performance (`R^2 = 0.9652`, `AARE = 9.09%`, `RMSE = 24.28 MPa`) is markedly better than both the conventional Arrhenius model and the pure neural network. In particular, the extrapolation AARE is reduced by about `49%` relative to Arrhenius and by more than `57%` relative to `NN-Direct`.

| Method | Dataset | R | R² | AARE (%) | RMSE (MPa) |
|---|---|---:|---:|---:|---:|
| Arrhenius | Training (strain rate <= 1 s^-1) | 0.9607 | 0.9229 | 23.14 | 24.82 |
| Arrhenius | Extrapolation (10 s^-1) | 0.9845 | 0.9693 | 17.84 | 41.32 |
| NN-Direct | Training (strain rate <= 1 s^-1) | 0.9999 | 0.9998 | 1.67 | 1.14 |
| NN-Direct | Extrapolation (10 s^-1) | 0.9479 | 0.8985 | 21.40 | 49.19 |
| NN-Arrhenius-Only | Training (strain rate <= 1 s^-1) | 0.9607 | 0.9230 | 23.04 | 24.83 |
| NN-Arrhenius-Only | Extrapolation (10 s^-1) | 0.9846 | 0.9693 | 17.84 | 41.31 |
| NN-PhysicsInit | Training (strain rate <= 1 s^-1) | 0.9959 | 0.9918 | 7.64 | 7.11 |
| NN-PhysicsInit | Extrapolation (10 s^-1) | 0.9824 | 0.9652 | 9.09 | 24.28 |

### 5.3 Mechanistic interpretation
The results from the two experiments jointly support a consistent interpretation.

First, the conventional Arrhenius model provides a strong global trend and therefore maintains stable extrapolation, but its limited expressive power prevents it from fully capturing complex local nonlinear behavior.

Second, `NN-Direct` is highly effective at interpolation but lacks sufficient physical regularization. As a result, it is more prone to overfitting and may exhibit unstable or oscillatory stress evolution when extrapolated to unseen conditions.

Third, `NN-Arrhenius-Only` performs very similarly to the Arrhenius baseline, which indicates that the neural network can indeed learn the physical structure embedded in the synthetic data. However, without real-data finetuning, it cannot surpass the baseline substantially.

Fourth, `NN-PhysicsInit` inherits the physically meaningful global trend from the Arrhenius pretraining stage and then adapts to experimental deviations through finetuning. This explains why it consistently achieves a superior compromise between accuracy and robustness.

## 6. Discussion
This work highlights an important point for constitutive modeling: the best interpolation model is not necessarily the best engineering model. In hot processing applications, extrapolation reliability is often more valuable than marginal gains in training accuracy. A purely data-driven neural network may look ideal when evaluated only on in-domain metrics, but such evaluation can conceal serious weaknesses outside the training range.

The present results suggest that physics priors should not be treated as optional interpretability add-ons. Instead, they can serve as an effective inductive bias that guides the network toward physically plausible solution manifolds before it sees scarce real data. This idea is especially important in materials science, where experiments are expensive and generalization to untested conditions is frequently required.

Another practical implication concerns metric design. The robust AARE used in the temperature extrapolation task shows that constitutive model evaluation must account for the physical meaning of low-stress regions. In such segments, small absolute deviations may produce disproportionately large relative errors. Therefore, engineering-oriented metrics can improve the fairness and usefulness of model comparison.

Despite the encouraging results, several limitations remain. The current study focuses on a single alloy system and uses stress prediction as the main target. The model does not yet explicitly incorporate microstructure evolution, phase transformation, or deformation mechanism partitioning. In addition, the synthetic pretraining data still depend on the quality of the initial Arrhenius fit; if the physics prior is systematically biased, the neural network may inherit part of that bias.

## 7. Engineering Value
The proposed framework offers direct engineering benefits.

1. It can support more reliable constitutive input for finite element simulations of hot forming.
2. It reduces the risk of unstable stress prediction under sparse test conditions or neighboring process windows.
3. It provides a practical route for extending constitutive models into untested temperature or strain-rate domains without fully abandoning physical consistency.
4. It offers a scalable methodology for digital manufacturing and intelligent process optimization in titanium alloys and other difficult-to-deform materials.

## 8. Conclusion
This study developed a physics-informed constitutive modeling framework for TC4 alloy by combining a strain-compensated Arrhenius model with neural network pretraining and finetuning. Two extrapolation scenarios, namely temperature extrapolation and strain-rate extrapolation, were designed to test the framework under practically meaningful conditions.

The results show that:

1. Purely data-driven neural networks achieve the best in-domain fitting but exhibit poor extrapolation robustness.
2. The Arrhenius model retains stable extrapolation behavior but lacks sufficient flexibility for complex nonlinear response.
3. The proposed `NN-PhysicsInit` effectively integrates the advantages of both routes and achieves the best overall trade-off between accuracy and robustness.
4. In temperature extrapolation, `NN-PhysicsInit` reduces the extrapolation AARE to `12.93%` and RMSE to `10.27 MPa`.
5. In strain-rate extrapolation, `NN-PhysicsInit` reduces the extrapolation AARE to `9.09%` and RMSE to `24.28 MPa`.

Overall, the study demonstrates that physics-prior initialization is a promising strategy for constitutive modeling under limited data and unseen deformation conditions. The framework provides a useful foundation for future extensions toward multi-task constitutive prediction, microstructure-aware modeling, and digital twin applications for hot processing.

## 9. Suggested Figure and Table Arrangement
### Figures
1. Overall framework of `NN-PhysicsInit`.
2. Distribution of synthetic data generated by the Arrhenius model.
3. Training loss comparison of different neural-network strategies.
4. Extrapolation stress-strain curves under temperature extrapolation.
5. Extrapolation stress-strain curves under strain-rate extrapolation.
6. Scatter plots comparing predicted and experimental stress.
7. Ablation bar charts of `R²`, `AARE`, and `RMSE`.
8. Oscillation analysis of `NN-Direct` in the extrapolation region.

### Tables
1. Experimental condition range and dataset split.
2. Hyperparameters of the neural networks.
3. Quantitative comparison under temperature extrapolation.
4. Quantitative comparison under strain-rate extrapolation.

## 10. Suggested References
以下参考文献为建议参考，正式投稿前请根据目标期刊格式逐条核对。

[1] Sellars, C. M., McTegart, W. J. On the mechanism of hot deformation. *Acta Metallurgica*, 1966, 14(9): 1136-1138.

[2] Solhjoo, S. Revisiting the common practice of Sellars and Tegart's hyperbolic sine constitutive model. *Modelling*, 2022, 3(3): 359-373.

[3] Cai, J., Li, F., Liu, T., Chen, B., Ma, F. Constitutive equations for elevated temperature flow stress of Ti-6Al-4V alloy considering the effect of strain. *Materials & Design*, 2011, 32(3): 1144-1151.

[4] Seshacharyulu, T., Medeiros, S. C., Morgan, J. T., Malas, J. C., Prasad, Y. V. R. K. Hot working of commercial Ti-6Al-4V with an equiaxed alpha-beta microstructure: materials modeling considerations. *Materials Science and Engineering A*, 2000, 284(1-2): 184-194.

[5] Souza, P. M., et al. Constitutive analysis of hot deformation behavior of a Ti-6Al-4V alloy using physical based model. *Materials Science and Engineering A*, 2015.

[6] Raissi, M., Perdikaris, P., Karniadakis, G. E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 2019, 378: 686-707.

[7] Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., Yang, L. Physics-informed machine learning. *Nature Reviews Physics*, 2021, 3: 422-440.

[8] Application of constitutive models and machine learning models to predict the elevated temperature flow behavior of TiAl alloy. *Materials*, 2023.

[9] Hot deformation and constitutive modeling of TC21 titanium alloy. *Materials*, 2022.

[10] Modified Fields-Backofen and Zerilli-Armstrong constitutive models to predict the hot deformation behavior in titanium-based alloys. *Scientific Reports*, 2024.

## 11. Notes for Next Revision
- 若你准备投稿英文期刊，下一步建议将本文稿整体转为英文并按目标期刊模板重排。
- 若你希望突出“方法创新”，可以将论文主标题聚焦在 `physics-prior-initialized neural network`。
- 若你希望突出“工程应用”，可以将论文主标题聚焦在 `robust extrapolation for hot processing design`。
- 若你后续还有显微组织、加工图或有限元模拟结果，可以自然扩展成更完整的 SCI 论文版本。
