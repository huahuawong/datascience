# Q1. What is A/B testing?
An experiment technique to determine if a new design brings improvement based on a given metric.
We should formulate A?B Testing using Population, Intervention, Comparison, Outcome, Time = PICOT


# Q2. Explain what is p value?
Before we talk about what is p value, we should understand what is null hypothesis. Null hypothesis is a hypothesis that states that when we
introduce a change to a system, there will be zero to none effects. If there really is no effect, we can declare that the null 
hypothesis is true, and p value evaluate how well our sample data support the null hypothesis.

High P values (typically > 0.05) means that data is likely with a true null hypothesis
Low P values (typically ≤ 0.05) means that data is unlikely with a true null hypothesis

# Q3. What is bias?
Bias is the dfference between the model prediction and the actual values that it is trying to predict. Bias tends to oversimplifies the model

# Q4. What is variance?
Variance is the variability of the model prediction given a testing dataset. Models with high variance tend to focus on training data, but not testing. This results in overfitting.

# Q5. What is the bias and variance tradeoff?
The need to strike a balance between these two, A good balance would ensure that the model isn’t underfitting or overfitting

# Q6. What’s confidence interval?
Let's say we want to have an idea of the population mean, but we can't be 100% certain on the actual number, so we get a range of values that we're confident that the actual value will be within this range

# Q7. What are the factors that affect confidence interval?
1. Sample size - higher sample size would result in wider CI and vice-versa
2. Variation - Low variation would result in narrower CI and vice-versa

# Q8. What is maximum likelihood estimation? Could there be any case where it doesn’t exist?
MLE is essentially finding a parameter which maximizes the likelihood that the process described by the model produced the data that were actually observed.
It can readily be generalized to the case where we are trying to estimate the probability density of observing a single data point x, that is generated from a Gaussian distribution, i.e. P(x; mean, standard deviation). It can be used to estimate parameters for machine learning models such as Logistic Regression

It doesn't exist when we want to model that with a mixture of Gaussians, where the estimation diverges. Let's say if we only have one Gaussian on one of the 
training examples. During MLE, the σ of the model will decrease, increasing the likelihood. As a result, other instances would become less likely.

# Q9. 
